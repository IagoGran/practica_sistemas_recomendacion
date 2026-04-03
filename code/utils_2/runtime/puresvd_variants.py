from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from global_utils.playlist_preprocessing import load_playlists_from_file
from utils_2.data.build_tracks_matrix import build_tracks_matrix
from utils_2.data.test_matrix_fixed_vocab import build_test_matrix_fixed_vocab
from utils_2.model.pure_svd_recommender import PureSVDRecommender
from utils_2.runtime.pure_svd_workers import run_puresvd_parallel


def _build_local_row_to_pid(input_playlists: List[dict]) -> Dict[int, int]:
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
    Build a lightweight model view with user factors restricted to test rows.
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
) -> Tuple[Dict[int, List[str]], float]:
    """
    Variant A:
    - fit on the joint train + test matrix
    - recommend only for test playlists
    """
    print("\n" + "=" * 70)
    print("EJECUTANDO PURESVD - VARIANTE A (fit train + test)")
    print("=" * 70)

    start_time = time.time()

    X_full, _, idx_to_track, playlist_id_to_row, _ = build_tracks_matrix(
        train_dir=train_dir,
        test_dir=test_dir,
        verbose=True,
    )

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

    test_rows = [playlist_id_to_row[playlist["pid"]] for playlist in input_playlists]
    X_query = X_full[test_rows]

    model_test = _build_test_view_model(
        model=model,
        test_rows=test_rows,
        X_query_shape=X_query.shape,
    )

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
) -> Tuple[Dict[int, List[str]], float]:
    """
    Variant B:
    - fit on train only
    - build test matrix with train vocabulary
    - recommend with folding-in
    """
    print("\n" + "=" * 70)
    print("EJECUTANDO PURESVD - VARIANTE B (fit train + folding-in)")
    print("=" * 70)

    start_time = time.time()

    X_train, track_to_idx, idx_to_track, _, _ = build_tracks_matrix(
        train_dir=train_dir,
        test_dir=None,
        verbose=True,
    )

    print(f"[B] Shape matriz train: {X_train.shape}")
    print(f"[B] NNZ: {X_train.nnz}")

    X_test, playlist_id_to_row_test, _ = build_test_matrix_fixed_vocab(
        test_dir=test_dir,
        track_to_idx=track_to_idx,
        verbose=True,
    )

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

    test_rows = [
        playlist_id_to_row_test[playlist["pid"]]
        for playlist in input_playlists
    ]
    X_query = X_test[test_rows]

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
        label="PureSVD-B",
    )

    elapsed = time.time() - start_time
    return results, elapsed
