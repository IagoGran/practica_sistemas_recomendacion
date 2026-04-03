from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, Optional, Tuple

from scipy.sparse import csr_matrix

from utils_2.model.pure_svd_recommender import PureSVDRecommender


G_SVD_MODEL = None
G_SVD_X_QUERY = None
G_SVD_ROW_TO_PID = None
G_SVD_IDX_TO_TRACK = None
G_SVD_TOP_K = 500
G_SVD_USE_FOLDING_IN = False
G_SVD_FILTER_SEEN = True
G_SVD_ITEM_BLOCK_SIZE = 50_000
G_SVD_VERBOSE = True


def _set_worker_state(
    model: PureSVDRecommender,
    X_query: csr_matrix,
    row_to_pid: Dict[int, int],
    idx_to_track: Dict[int, str],
    top_k: int,
    use_folding_in: bool,
    filter_seen: bool,
    item_block_size: Optional[int],
    verbose: bool,
) -> None:
    """
    Store the shared read-only objects consumed by recommendation workers.

    Parameters
    ----------
    model:
        Fitted PureSVD model used to score the queried playlists.
    X_query:
        Sparse query matrix shared by every worker.
    row_to_pid:
        Mapping from local row offset to playlist identifier.
    idx_to_track:
        Mapping from item index to track URI.
    top_k:
        Number of recommendations to keep per playlist.
    use_folding_in:
        Whether the query users must be projected with folding-in.
    filter_seen:
        Whether already-seen tracks must be excluded from the ranking.
    item_block_size:
        Number of catalog items scored at once inside the recommender.
    verbose:
        Whether workers should emit progress information.
    """
    global G_SVD_MODEL, G_SVD_X_QUERY, G_SVD_ROW_TO_PID, G_SVD_IDX_TO_TRACK
    global G_SVD_TOP_K, G_SVD_USE_FOLDING_IN, G_SVD_FILTER_SEEN
    global G_SVD_ITEM_BLOCK_SIZE, G_SVD_VERBOSE

    G_SVD_MODEL = model
    G_SVD_X_QUERY = X_query
    G_SVD_ROW_TO_PID = row_to_pid
    G_SVD_IDX_TO_TRACK = idx_to_track
    G_SVD_TOP_K = top_k
    G_SVD_USE_FOLDING_IN = use_folding_in
    G_SVD_FILTER_SEEN = filter_seen
    G_SVD_ITEM_BLOCK_SIZE = item_block_size
    G_SVD_VERBOSE = verbose


def init_worker_puresvd(
    model: PureSVDRecommender,
    X_query: csr_matrix,
    row_to_pid: Dict[int, int],
    idx_to_track: Dict[int, str],
    top_k: int,
    use_folding_in: bool,
    filter_seen: bool,
    item_block_size: Optional[int],
    verbose: bool,
) -> None:
    """
    Initialize the process-local state needed by PureSVD workers.

    Parameters
    ----------
    model:
        Fitted PureSVD model used to score the queried playlists.
    X_query:
        Sparse query matrix shared by every worker.
    row_to_pid:
        Mapping from local row offset to playlist identifier.
    idx_to_track:
        Mapping from item index to track URI.
    top_k:
        Number of recommendations to keep per playlist.
    use_folding_in:
        Whether the query users must be projected with folding-in.
    filter_seen:
        Whether already-seen tracks must be excluded from the ranking.
    item_block_size:
        Number of catalog items scored at once inside the recommender.
    verbose:
        Whether workers should emit progress information.
    """
    _set_worker_state(
        model=model,
        X_query=X_query,
        row_to_pid=row_to_pid,
        idx_to_track=idx_to_track,
        top_k=top_k,
        use_folding_in=use_folding_in,
        filter_seen=filter_seen,
        item_block_size=item_block_size,
        verbose=verbose,
    )


def chunk_row_ranges(n_rows: int, chunk_size: int) -> List[Tuple[int, int]]:
    """
    Split the query rows into contiguous chunks.

    Parameters
    ----------
    n_rows:
        Total number of query rows.
    chunk_size:
        Maximum number of rows per chunk.

    Returns
    -------
    List[Tuple[int, int]]
        List of ``(start, end)`` row ranges.
    """
    if n_rows <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0")

    return [
        (start, min(start + chunk_size, n_rows))
        for start in range(0, n_rows, chunk_size)
    ]


def process_row_chunk_puresvd(row_range: Tuple[int, int]) -> Dict[int, List[str]]:
    """
    Recommend tracks for one row chunk and convert indices to track URIs.

    Parameters
    ----------
    row_range:
        Inclusive-exclusive row range assigned to the worker.

    Returns
    -------
    Dict[int, List[str]]
        Mapping from playlist id to the ordered list of recommended tracks.
    """
    start, end = row_range
    X_block = G_SVD_X_QUERY[start:end]

    block_result = G_SVD_MODEL.recommend_block(
        start=start,
        end=end,
        X_query_block=X_block,
        top_k=G_SVD_TOP_K,
        use_folding_in=G_SVD_USE_FOLDING_IN,
        filter_seen=G_SVD_FILTER_SEEN,
        item_block_size=G_SVD_ITEM_BLOCK_SIZE,
    )

    results: Dict[int, List[str]] = {}

    for local_row, global_row in enumerate(range(start, end)):
        pid = G_SVD_ROW_TO_PID[global_row]
        track_indices = block_result.top_indices[local_row]
        recommendations = [G_SVD_IDX_TO_TRACK[int(idx)] for idx in track_indices]

        if len(recommendations) != G_SVD_TOP_K:
            raise ValueError(
                f"[PureSVD] pid {pid}: esperado {G_SVD_TOP_K}, generado {len(recommendations)}"
            )
        if len(set(recommendations)) != G_SVD_TOP_K:
            raise ValueError(f"[PureSVD] pid {pid}: duplicados en recomendaciones")

        results[pid] = recommendations

    return results


def run_puresvd_parallel(
    model: PureSVDRecommender,
    X_query: csr_matrix,
    row_to_pid: Dict[int, int],
    idx_to_track: Dict[int, str],
    num_workers: int = 4,
    chunk_size: int = 250,
    top_k: int = 500,
    use_folding_in: bool = False,
    filter_seen: bool = True,
    item_block_size: Optional[int] = 50_000,
    parallel_backend: Literal["thread", "process"] = "thread",
    verbose: bool = True,
    label: str = "PureSVD",
) -> Dict[int, List[str]]:
    """
    Execute block-wise PureSVD recommendation in parallel.

    Parameters
    ----------
    model:
        Fitted PureSVD model ready to score playlists.
    X_query:
        Sparse query matrix whose rows will be recommended.
    row_to_pid:
        Mapping from local row offset to playlist id.
    idx_to_track:
        Mapping from item index to track URI.
    num_workers:
        Number of worker threads or processes.
    chunk_size:
        Number of playlists processed by each worker task.
    top_k:
        Number of recommendations requested per playlist.
    use_folding_in:
        Whether the user factors must be projected from ``X_query``.
    filter_seen:
        Whether already-seen tracks must be excluded.
    item_block_size:
        Number of catalog items scored at once inside the recommender.
    parallel_backend:
        Execution backend. ``thread`` avoids copying large matrices in memory.
    verbose:
        Whether to print execution progress.
    label:
        Prefix used in progress logs.

    Returns
    -------
    Dict[int, List[str]]
        Mapping from playlist id to recommended track URIs.
    """
    if not isinstance(X_query, csr_matrix):
        raise TypeError("X_query debe ser una csr_matrix")
    if X_query.shape[0] != len(row_to_pid):
        raise ValueError("row_to_pid debe cubrir todas las filas de X_query")
    if parallel_backend not in {"thread", "process"}:
        raise ValueError("parallel_backend debe ser 'thread' o 'process'")

    n_rows = X_query.shape[0]
    row_chunks = chunk_row_ranges(n_rows, chunk_size)

    if verbose:
        print(
            f"[{label}] chunks: {len(row_chunks)} | "
            f"chunk_size: {chunk_size} | workers: {num_workers} | "
            f"backend: {parallel_backend} | item_block_size: {item_block_size}"
        )

    results: Dict[int, List[str]] = {}
    start_time = time.time()
    completed_chunks = 0
    total_chunks = len(row_chunks)

    if parallel_backend == "thread":
        # Threads share the same address space, avoiding huge model/matrix copies.
        _set_worker_state(
            model=model,
            X_query=X_query,
            row_to_pid=row_to_pid,
            idx_to_track=idx_to_track,
            top_k=top_k,
            use_folding_in=use_folding_in,
            filter_seen=filter_seen,
            item_block_size=item_block_size,
            verbose=verbose,
        )
        executor_cm = ThreadPoolExecutor(max_workers=num_workers)
    else:
        executor_cm = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker_puresvd,
            initargs=(
                model,
                X_query,
                row_to_pid,
                idx_to_track,
                top_k,
                use_folding_in,
                filter_seen,
                item_block_size,
                verbose,
            ),
        )

    with executor_cm as executor:
        futures = [
            executor.submit(process_row_chunk_puresvd, row_range)
            for row_range in row_chunks
        ]

        for future in as_completed(futures):
            partial = future.result()
            results.update(partial)

            completed_chunks += 1
            processed = min(completed_chunks * chunk_size, n_rows)
            elapsed = time.time() - start_time
            avg_per_playlist = elapsed / processed if processed > 0 else 0.0
            remaining = avg_per_playlist * (n_rows - processed)

            if verbose:
                print(
                    f"[{label}] {completed_chunks}/{total_chunks} | "
                    f"{processed}/{n_rows} playlists | "
                    f"{elapsed:.2f}s transcurridos | "
                    f"{remaining / 60:.2f} min restantes"
                )

    return results
