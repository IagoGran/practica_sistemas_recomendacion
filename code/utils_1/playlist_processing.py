from collections import defaultdict
import json
import os
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix
from global_utils.playlist_preprocessing import iter_playlists_from_dir


def build_tracks_matrix(
                        train_dir: str
                        ) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str], Dict[int, int], Dict[int, int]]:
    """
    Matriz Playlist-Track:
      - filas = playlists
      - columnas = tracks
      - valor = 1 si el track aparece en la playlist
    Devuelve: (matrix, track_to_idx, idx_to_track, playlist_id_to_row, row_to_playlist_id)
    - matrix: csr_matrix de shape (n_playlists, n_tracks)
    - track_to_idx: dict track_uri -> columna
    - idx_to_track: dict columna -> track_uri
    - playlist_id_to_row: dict playlist_id -> fila
    - row_to_playlist_id: dict fila -> playlist_id
    """
    track_to_idx: Dict[str, int] = {}
    playlist_id_to_row: Dict[int, int] = {}
    idx_to_track: Dict[int, str] = {}
    row_to_playlist_id: Dict[int, int] = {}

    rows = []
    cols = []
    values = []

    for pl in iter_playlists_from_dir(train_dir):
        playlist_id = pl.get("pid")
        if playlist_id is None:
            continue

        # asigna índice de fila a cada playlist
        row = playlist_id_to_row.get(playlist_id)
        if row is None:
            row = len(playlist_id_to_row)
            playlist_id_to_row[playlist_id] = row
            row_to_playlist_id[row] = playlist_id

        # para evitar duplicados dentro de la misma playlist
        seen_in_playlist = set()

        for tr in pl.get("tracks", []):
            uri = tr.get("track_uri")
            if not uri or uri in seen_in_playlist:
                continue
            seen_in_playlist.add(uri)

            col = track_to_idx.get(uri)
            if col is None:
                col = len(track_to_idx)
                track_to_idx[uri] = col
                idx_to_track[col] = uri

            rows.append(row)
            cols.append(col)
            values.append(1)

    n_playlists = len(playlist_id_to_row)
    n_tracks = len(track_to_idx)

    X = csr_matrix(
    (np.array(values, dtype=np.int32), (np.array(rows), np.array(cols))),
    shape=(n_playlists, n_tracks),
    dtype=np.int32
    )
    X.sum_duplicates()  # por si acaso


    return X, track_to_idx, idx_to_track, playlist_id_to_row, row_to_playlist_id

def popularity_from_matrix(X: csr_matrix, track_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Dada la matriz Playlist-Track, devuelve una lista de (track_uri, count) ordenada por count desc.
    """
    track_popularity = np.array(X.sum(axis=0)).flatten()  # suma por columnas
    idx_to_track = {idx: uri for uri, idx in track_to_idx.items()}
    popularity_list = [(idx_to_track[idx], count) for idx, count in enumerate(track_popularity)]
    # ordenamos por count desc
    popularity_list.sort(key=lambda x: x[1], reverse=True)
    return popularity_list

def build_global_popularity(
                            train_dir: str
                            ) -> Tuple[
                                csr_matrix,
                                Dict[str, int],
                                Dict[int, str],
                                Dict[int, int],
                                Dict[int, int],
                                List[Tuple[str, int]]
                            ]:
    X, track_to_idx, idx_to_track, playlist_id_to_row, row_to_playlist_id = build_tracks_matrix(train_dir)
    popularity_list = popularity_from_matrix(X, track_to_idx)
    return X, track_to_idx, idx_to_track, playlist_id_to_row, row_to_playlist_id, popularity_list

def _seed_tracks_to_indices(
    seed_tracks: Set[str],
    track_to_idx: Dict[str, int]
) -> List[int]:
    return [track_to_idx[t] for t in seed_tracks if t in track_to_idx]

def _collect_playlist_overlaps(
    seed_track_idxs: List[int],
    XT: csr_matrix,
    max_playlist_freq_per_seed_track: Optional[int]
) -> Dict[int, int]:

    playlist_overlap: Dict[int, int] = defaultdict(int)

    for track_idx in seed_track_idxs:
        candidate_rows = XT[track_idx].indices

        if (
            max_playlist_freq_per_seed_track is not None
            and len(candidate_rows) > max_playlist_freq_per_seed_track
        ):
            continue

        for row in candidate_rows:
            playlist_overlap[row] += 1

    return playlist_overlap

def _compute_candidate_similarities(
    playlist_overlap: Dict[int, int],
    playlist_norms: np.ndarray,
    seed_len: int
) -> Tuple[np.ndarray, np.ndarray]:

    seed_norm = np.sqrt(seed_len).astype(np.float32)

    candidate_rows = np.fromiter(playlist_overlap.keys(), dtype=np.int32)
    overlaps = np.fromiter(playlist_overlap.values(), dtype=np.float32)

    denom = seed_norm * playlist_norms[candidate_rows]
    valid = denom > 0

    if not np.any(valid):
        return np.array([]), np.array([])

    candidate_rows = candidate_rows[valid]
    overlaps = overlaps[valid]

    sims = overlaps / denom[valid]

    return candidate_rows, sims

def _select_top_neighbors(
    candidate_rows: np.ndarray,
    sims: np.ndarray,
    top_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:

    if len(candidate_rows) > top_neighbors:
        idx = np.argpartition(sims, -top_neighbors)[-top_neighbors:]
        candidate_rows = candidate_rows[idx]
        sims = sims[idx]

    order = np.argsort(sims)[::-1]

    return candidate_rows[order], sims[order]

def _accumulate_track_scores(
    candidate_rows: np.ndarray,
    sims: np.ndarray,
    X: csr_matrix,
    seed_track_idx_set: Set[int]
) -> Dict[int, float]:

    track_scores: Dict[int, float] = defaultdict(float)

    for row, sim in zip(candidate_rows, sims):

        track_indices = X[row].indices

        for track_idx in track_indices:
            if track_idx not in seed_track_idx_set:
                track_scores[track_idx] += float(sim)

    return track_scores

def _build_recommendations_from_scores(
    track_scores: Dict[int, float],
    idx_to_track: Dict[int, str],
    seed_tracks: Set[str],
    k: int
) -> Tuple[List[str], Set[str]]:

    recs: List[str] = []
    already_added: Set[str] = set()

    ranked = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)

    for track_idx, _ in ranked:

        track_uri = idx_to_track[track_idx]

        if track_uri not in seed_tracks and track_uri not in already_added:
            recs.append(track_uri)
            already_added.add(track_uri)

        if len(recs) >= k:
            break

    return recs, already_added

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

def recommend_for_seed_playlist_fast(
    seed_tracks: Set[str],
    track_to_idx: Dict[str, int],
    idx_to_track: Dict[int, str],
    popularity_list: List[Tuple[str, int]],
    X: csr_matrix,
    XT: csr_matrix,
    playlist_norms: np.ndarray,
    k: int = 500,
    top_neighbors: int = 1000,
    max_playlist_freq_per_seed_track: Optional[int] = 50000
) -> List[str]:

    seed_track_idxs = _seed_tracks_to_indices(seed_tracks, track_to_idx)

    if not seed_track_idxs:
        return _fill_with_popularity([], set(), seed_tracks, popularity_list, k)

    seed_track_idx_set = set(seed_track_idxs)

    playlist_overlap = _collect_playlist_overlaps(
        seed_track_idxs,
        XT,
        max_playlist_freq_per_seed_track
    )

    if not playlist_overlap:
        return _fill_with_popularity([], set(), seed_tracks, popularity_list, k)

    candidate_rows, sims = _compute_candidate_similarities(
        playlist_overlap,
        playlist_norms,
        len(seed_track_idxs)
    )

    if len(candidate_rows) == 0:
        return _fill_with_popularity([], set(), seed_tracks, popularity_list, k)

    candidate_rows, sims = _select_top_neighbors(
        candidate_rows,
        sims,
        top_neighbors
    )

    track_scores = _accumulate_track_scores(
        candidate_rows,
        sims,
        X,
        seed_track_idx_set
    )

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