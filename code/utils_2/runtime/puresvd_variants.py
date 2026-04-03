from __future__ import annotations

import time
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from global_utils.playlist_preprocessing import load_playlists_from_file
from utils_2.data.build_tracks_matrix import build_tracks_matrix
from utils_2.data.test_matrix_fixed_vocab import build_test_matrix_fixed_vocab
from utils_2.model.pure_svd_recommender import PureSVDRecommender
from utils_2.runtime.pure_svd_workers import run_puresvd_parallel


def _build_local_row_to_pid(input_playlists: List[dict]) -> Dict[int, int]:
    """
    Build the local row-to-playlist-id mapping used by the query matrix.

    Parameters
    ----------
    input_playlists:
        Ordered playlists that define the row order of the query matrix.

    Returns
    -------
    Dict[int, int]
        Mapping from local row index to playlist id.
    """
    return {
        row_idx: playlist["pid"]
        for row_idx, playlist in enumerate(input_playlists)
    }


def _build_test_view_model(
    model: PureSVDRecommender,
    test_rows: List[int],
    X_query_shape: Tuple[int, int],
) -> PureSVDRecommender:
    """
    Build a lightweight model view restricted to the queried test playlists.

    Parameters
    ----------
    model:
        Fitted model trained on the complete joint matrix.
    test_rows:
        Row indices of the queried test playlists inside the original matrix.
    X_query_shape:
        Shape of the test-only query matrix.

    Returns
    -------
    PureSVDRecommender
        View of the original model whose user factors contain only the queried
        test playlists.
    """
    model_test = PureSVDRecommender(
        n_factors=model.n_factors,
        random_state=model.random_state,
        dtype=model.dtype,
    )

    model_test.user_factors_ = model.user_factors_[test_rows]
    model_test.s_ = model.s_
    model_test.s_inv_ = model.s_inv_
    model_test.vt_ = model.vt_
    model_test.v_ = model.v_
    model_test.item_factors_ = model.item_factors_
    model_test.train_shape_ = X_query_shape
    model_test.n_users_ = X_query_shape[0]
    model_test.n_items_ = model.n_items_
    model_test.effective_n_factors_ = model.effective_n_factors_
    model_test.is_fitted_ = True

    return model_test


def run_variant_a(
    train_dir: str,
    test_dir: str,
    test_input_file: str,
    input_playlists: Optional[List[dict]] = None,
    num_factors: int = 128,
    num_workers: int = 4,
    chunk_size: int = 250,
    top_k: int = 500,
    item_block_size: Optional[int] = 50_000,
    parallel_backend: Literal["thread", "process"] = "thread",
    verbose: bool = True,
) -> Tuple[Dict[int, List[str]], float]:
    """
    Execute PureSVD variant A over the iteration-2 dataset.

    Variant A trains on the joint train+test matrix and then recommends only
    for the playlists that belong to the test split.

    Parameters
    ----------
    train_dir:
        Directory containing the training playlist JSON files.
    test_dir:
        Directory containing the test playlist JSON files.
    test_input_file:
        JSON file with the ordered test playlists used for querying.
    input_playlists:
        Optional preloaded version of ``test_input_file``.
    num_factors:
        Latent dimensionality used by PureSVD.
    num_workers:
        Number of worker threads or processes.
    chunk_size:
        Number of playlists processed by each worker task.
    top_k:
        Number of recommendations requested per playlist.
    item_block_size:
        Number of catalog items scored at once inside the recommender.
    parallel_backend:
        Execution backend used by the parallel recommender.
    verbose:
        Whether to print execution progress.

    Returns
    -------
    Tuple[Dict[int, List[str]], float]
        Pair ``(results, elapsed_seconds)``.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EJECUTANDO PURESVD - VARIANTE A (fit train + test)")
        print("=" * 70)

    start_time = time.time()

    X_full, _, idx_to_track, playlist_id_to_row, _ = build_tracks_matrix(
        train_dir=train_dir,
        test_dir=test_dir,
        verbose=verbose,
    )

    if verbose:
        print(f"[A] Shape matriz conjunta: {X_full.shape}")
        print(f"[A] NNZ: {X_full.nnz}")

    model = PureSVDRecommender(
        n_factors=num_factors,
        random_state=42,
        dtype=np.float32,
    )
    model.fit(X_full)

    if input_playlists is None:
        input_playlists = load_playlists_from_file(test_input_file)

    if verbose:
        print("[A] Reordenando playlists de test segun test_input_file...")

    test_rows = [playlist_id_to_row[playlist["pid"]] for playlist in input_playlists]
    X_query = X_full[test_rows]

    if verbose:
        print("[A] Construyendo vista del modelo restringida a las filas de test...")

    model_test = _build_test_view_model(
        model=model,
        test_rows=test_rows,
        X_query_shape=X_query.shape,
    )

    if verbose:
        print("[A] Lanzando recomendacion paralela...")

    results = run_puresvd_parallel(
        model=model_test,
        X_query=X_query,
        row_to_pid=_build_local_row_to_pid(input_playlists),
        idx_to_track=idx_to_track,
        num_workers=num_workers,
        chunk_size=chunk_size,
        top_k=top_k,
        use_folding_in=False,
        filter_seen=True,
        item_block_size=item_block_size,
        parallel_backend=parallel_backend,
        verbose=verbose,
        label="PureSVD-A",
    )

    elapsed = time.time() - start_time
    return results, elapsed


def run_variant_b(
    train_dir: str,
    test_dir: str,
    test_input_file: str,
    input_playlists: Optional[List[dict]] = None,
    num_factors: int = 128,
    num_workers: int = 4,
    chunk_size: int = 250,
    top_k: int = 500,
    item_block_size: Optional[int] = 50_000,
    parallel_backend: Literal["thread", "process"] = "thread",
    verbose: bool = True,
) -> Tuple[Dict[int, List[str]], float]:
    """
    Execute PureSVD variant B over the iteration-2 dataset.

    Variant B trains only on the train split, projects the test playlists with
    folding-in and recommends using the train vocabulary.

    Parameters
    ----------
    train_dir:
        Directory containing the training playlist JSON files.
    test_dir:
        Directory containing the test playlist JSON files.
    test_input_file:
        JSON file with the ordered test playlists used for querying.
    input_playlists:
        Optional preloaded version of ``test_input_file``.
    num_factors:
        Latent dimensionality used by PureSVD.
    num_workers:
        Number of worker threads or processes.
    chunk_size:
        Number of playlists processed by each worker task.
    top_k:
        Number of recommendations requested per playlist.
    item_block_size:
        Number of catalog items scored at once inside the recommender.
    parallel_backend:
        Execution backend used by the parallel recommender.
    verbose:
        Whether to print execution progress.

    Returns
    -------
    Tuple[Dict[int, List[str]], float]
        Pair ``(results, elapsed_seconds)``.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("EJECUTANDO PURESVD - VARIANTE B (fit train + folding-in)")
        print("=" * 70)

    start_time = time.time()

    X_train, track_to_idx, idx_to_track, _, _ = build_tracks_matrix(
        train_dir=train_dir,
        test_dir=None,
        verbose=verbose,
    )

    if verbose:
        print(f"[B] Shape matriz train: {X_train.shape}")
        print(f"[B] NNZ: {X_train.nnz}")

    X_test, playlist_id_to_row_test, _ = build_test_matrix_fixed_vocab(
        test_dir=test_dir,
        track_to_idx=track_to_idx,
        verbose=verbose,
    )

    if verbose:
        print(f"[B] Shape matriz test fixed vocab: {X_test.shape}")
        print(f"[B] NNZ test: {X_test.nnz}")

    model = PureSVDRecommender(
        n_factors=num_factors,
        random_state=42,
        dtype=np.float32,
    )
    model.fit(X_train)

    if input_playlists is None:
        input_playlists = load_playlists_from_file(test_input_file)

    if verbose:
        print("[B] Reordenando playlists de test segun test_input_file...")

    test_rows = [
        playlist_id_to_row_test[playlist["pid"]]
        for playlist in input_playlists
    ]
    X_query = X_test[test_rows]

    if verbose:
        print("[B] Lanzando recomendacion paralela con folding-in...")

    results = run_puresvd_parallel(
        model=model,
        X_query=X_query,
        row_to_pid=_build_local_row_to_pid(input_playlists),
        idx_to_track=idx_to_track,
        num_workers=num_workers,
        chunk_size=chunk_size,
        top_k=top_k,
        use_folding_in=True,
        filter_seen=True,
        item_block_size=item_block_size,
        parallel_backend=parallel_backend,
        verbose=verbose,
        label="PureSVD-B",
    )

    elapsed = time.time() - start_time
    return results, elapsed
