from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import heapq

# ==========================
# utilidades básicas
# ==========================

def compute_track_frequencies(X: csr_matrix) -> np.ndarray:
    return np.asarray(X.sum(axis=0)).ravel().astype(np.int32)


def _seed_tracks_to_indices(
    seed_tracks: Set[str],
    track_to_idx: Dict[str, int]
) -> List[int]:
    return [track_to_idx[t] for t in seed_tracks if t in track_to_idx]


def _compute_seed_weight(
    seed_track_idx: int,
    track_freqs: np.ndarray,
    n_playlists: int
) -> float:

    df = track_freqs[seed_track_idx]
    if df <= 0:
        return 0.0

    return float(np.log1p(n_playlists / df))


def _fill_with_popularity(
    recs: List[str],
    already_added: Set[str],
    seed_tracks: Set[str],
    popularity_list: List[Tuple[str, int]],
    k: int
) -> List[str]:

    for track_uri, _ in popularity_list:

        if track_uri not in seed_tracks and track_uri not in already_added:

            recs.append(track_uri)
            already_added.add(track_uri)

        if len(recs) >= k:
            break

    return recs


# ==========================
# extracción de seeds
# ==========================

def extract_unique_seed_track_idxs(
    input_playlists: List[dict],
    track_to_idx: Dict[str, int]
) -> List[int]:

    unique_seed_track_idxs: Set[int] = set()

    for pl in input_playlists:

        for tr in pl.get("tracks", []):

            track_uri = tr.get("track_uri")

            if track_uri is not None and track_uri in track_to_idx:

                unique_seed_track_idxs.add(track_to_idx[track_uri])

    return sorted(unique_seed_track_idxs)


# ==========================
# cálculo coocurrencias
# ==========================

def _collect_cooccurring_tracks_for_seed(
    seed_track_idx: int,
    X: csr_matrix,
    XT: csr_matrix,
    max_playlists_per_seed_track: Optional[int],
    min_cooccurrence: int = 2
) -> Dict[int, int]:

    cooc_counts: Dict[int, int] = defaultdict(int)

    playlist_rows = XT[seed_track_idx].indices

    if (
        max_playlists_per_seed_track is not None
        and len(playlist_rows) > max_playlists_per_seed_track
    ):

        rng = np.random.default_rng(seed_track_idx)

        playlist_rows = rng.choice(
            playlist_rows,
            size=max_playlists_per_seed_track,
            replace=False
        )

    for row in playlist_rows:

        track_indices = X[row].indices

        for track_idx in track_indices:

            if track_idx != seed_track_idx:

                cooc_counts[track_idx] += 1

    if min_cooccurrence > 1:

        cooc_counts = {
            track_idx: count
            for track_idx, count in cooc_counts.items()
            if count >= min_cooccurrence
        }

    return cooc_counts


# ==========================
# similitud coseno binaria
# ==========================

def _compute_track_to_track_scores(
    seed_track_idx: int,
    cooc_counts: Dict[int, int],
    track_freqs: np.ndarray,
    min_similarity: float = 0.0
) -> Dict[int, float]:

    scores: Dict[int, float] = {}

    df_i = track_freqs[seed_track_idx]

    if df_i <= 0:
        return scores

    for other_track_idx, cooc in cooc_counts.items():

        df_j = track_freqs[other_track_idx]

        if df_j <= 0:
            continue

        denom = np.sqrt(df_i * df_j)

        if denom <= 0:
            continue

        sim = float(cooc) / float(denom)

        if sim >= min_similarity:

            scores[other_track_idx] = sim

    return scores


# ==========================
# vecinos para un seed
# ==========================

def precompute_neighbors_for_seed_track(
    seed_track_idx: int,
    X: csr_matrix,
    XT: csr_matrix,
    track_freqs: np.ndarray,
    max_playlists_per_seed_track: Optional[int] = 2000,
    min_cooccurrence: int = 2,
    min_similarity: float = 0.01,
    top_k_per_seed_track: Optional[int] = 100
) -> List[Tuple[int, float]]:

    cooc_counts = _collect_cooccurring_tracks_for_seed(
        seed_track_idx,
        X,
        XT,
        max_playlists_per_seed_track,
        min_cooccurrence
    )

    if not cooc_counts:
        return []

    local_scores = _compute_track_to_track_scores(
        seed_track_idx,
        cooc_counts,
        track_freqs,
        min_similarity
    )

    if not local_scores:
        return []

    if top_k_per_seed_track is not None:

        ranked_local = heapq.nlargest(
            top_k_per_seed_track,
            local_scores.items(),
            key=lambda x: x[1]
        )

    else:

        ranked_local = sorted(
            local_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

    return ranked_local


# ==========================
# paralelización precompute
# ==========================

GP_X = None
GP_XT = None
GP_TRACK_FREQS = None
GP_MAX_PLAYLISTS_PER_SEED_TRACK = None
GP_MIN_COOCCURRENCE = None
GP_MIN_SIMILARITY = None
GP_TOP_K_PER_SEED_TRACK = None


def init_worker_precompute(
    X,
    XT,
    track_freqs,
    max_playlists_per_seed_track,
    min_cooccurrence,
    min_similarity,
    top_k_per_seed_track
):

    global GP_X, GP_XT, GP_TRACK_FREQS
    global GP_MAX_PLAYLISTS_PER_SEED_TRACK
    global GP_MIN_COOCCURRENCE
    global GP_MIN_SIMILARITY
    global GP_TOP_K_PER_SEED_TRACK

    GP_X = X
    GP_XT = XT
    GP_TRACK_FREQS = track_freqs
    GP_MAX_PLAYLISTS_PER_SEED_TRACK = max_playlists_per_seed_track
    GP_MIN_COOCCURRENCE = min_cooccurrence
    GP_MIN_SIMILARITY = min_similarity
    GP_TOP_K_PER_SEED_TRACK = top_k_per_seed_track


def process_seed_chunk(seed_chunk: List[int]):

    partial: Dict[int, List[Tuple[int, float]]] = {}

    for seed_track_idx in seed_chunk:

        neighbors = precompute_neighbors_for_seed_track(
            seed_track_idx,
            GP_X,
            GP_XT,
            GP_TRACK_FREQS,
            GP_MAX_PLAYLISTS_PER_SEED_TRACK,
            GP_MIN_COOCCURRENCE,
            GP_MIN_SIMILARITY,
            GP_TOP_K_PER_SEED_TRACK
        )

        partial[seed_track_idx] = neighbors

    return partial


# ==========================
# recomendación
# ==========================

def _build_recommendations_from_scores(
    track_scores: Dict[int, float],
    idx_to_track: Dict[int, str],
    seed_tracks: Set[str],
    k: int
):

    recs: List[str] = []
    already_added: Set[str] = set()

    ranked = heapq.nlargest(k, track_scores.items(), key=lambda x: x[1])

    for track_idx, _ in ranked:

        track_uri = idx_to_track[track_idx]

        if track_uri not in seed_tracks and track_uri not in already_added:

            recs.append(track_uri)
            already_added.add(track_uri)

        if len(recs) >= k:
            break

    return recs, already_added


def recommend_from_precomputed_neighbors(
    seed_tracks: Set[str],
    track_to_idx: Dict[str, int],
    idx_to_track: Dict[int, str],
    popularity_list: List[Tuple[str, int]],
    precomputed_neighbors: Dict[int, List[Tuple[int, float]]],
    track_freqs: np.ndarray,
    n_playlists: int,
    k: int = 500,
    use_seed_idf_weight: bool = True
):

    seed_track_idxs = _seed_tracks_to_indices(seed_tracks, track_to_idx)

    if not seed_track_idxs:

        return _fill_with_popularity([], set(), seed_tracks, popularity_list, k)

    seed_track_idx_set = set(seed_track_idxs)

    track_scores: Dict[int, float] = defaultdict(float)

    for seed_track_idx in seed_track_idxs:

        neighbors = precomputed_neighbors.get(seed_track_idx)

        if not neighbors:
            continue

        seed_weight = 1.0

        if use_seed_idf_weight:

            seed_weight = _compute_seed_weight(
                seed_track_idx,
                track_freqs,
                n_playlists
            )

        for neighbor_track_idx, score in neighbors:

            if neighbor_track_idx not in seed_track_idx_set:

                track_scores[neighbor_track_idx] += seed_weight * score

    recs, already_added = _build_recommendations_from_scores(
        track_scores,
        idx_to_track,
        seed_tracks,
        k
    )

    if len(recs) < k:

        recs = _fill_with_popularity(
            recs,
            already_added,
            seed_tracks,
            popularity_list,
            k
        )

    return recs