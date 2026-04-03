from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from scipy.sparse import csr_matrix

from utils_2.model.pure_svd_recommender import PureSVDRecommender


G_SVD_MODEL = None
G_SVD_X_QUERY = None
G_SVD_ROW_TO_PID = None
G_SVD_IDX_TO_TRACK = None
G_SVD_TOP_K = 500
G_SVD_USE_FOLDING_IN = False
G_SVD_FILTER_SEEN = True


def init_worker_puresvd(
    model: PureSVDRecommender,
    X_query: csr_matrix,
    row_to_pid: Dict[int, int],
    idx_to_track: Dict[int, str],
    top_k: int,
    use_folding_in: bool,
    filter_seen: bool,
) -> None:
    """
    Initialize the global worker state for PureSVD recommendation.
    """
    global G_SVD_MODEL, G_SVD_X_QUERY, G_SVD_ROW_TO_PID, G_SVD_IDX_TO_TRACK
    global G_SVD_TOP_K, G_SVD_USE_FOLDING_IN, G_SVD_FILTER_SEEN

    G_SVD_MODEL = model
    G_SVD_X_QUERY = X_query
    G_SVD_ROW_TO_PID = row_to_pid
    G_SVD_IDX_TO_TRACK = idx_to_track
    G_SVD_TOP_K = top_k
    G_SVD_USE_FOLDING_IN = use_folding_in
    G_SVD_FILTER_SEEN = filter_seen


def chunk_row_ranges(n_rows: int, chunk_size: int) -> List[Tuple[int, int]]:
    """
    Split row indices into (start, end) chunks.
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
    Recommend tracks for a row chunk and return {pid: recommendations}.
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
    label: str = "PureSVD",
) -> Dict[int, List[str]]:
    """
    Run block-wise PureSVD recommendation in parallel.
    """
    if not isinstance(X_query, csr_matrix):
        raise TypeError("X_query debe ser una csr_matrix")
    if X_query.shape[0] != len(row_to_pid):
        raise ValueError("row_to_pid debe cubrir todas las filas de X_query")

    n_rows = X_query.shape[0]
    row_chunks = chunk_row_ranges(n_rows, chunk_size)

    print(
        f"[{label}] chunks: {len(row_chunks)} | "
        f"chunk_size: {chunk_size} | workers: {num_workers}"
    )

    results: Dict[int, List[str]] = {}
    start_time = time.time()
    completed_chunks = 0
    total_chunks = len(row_chunks)

    with ProcessPoolExecutor(
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
        ),
    ) as executor:
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

            print(
                f"[{label}] {completed_chunks}/{total_chunks} | "
                f"{processed}/{n_rows} playlists | "
                f"{elapsed:.2f}s transcurridos | "
                f"{remaining / 60:.2f} min restantes"
            )

    return results

