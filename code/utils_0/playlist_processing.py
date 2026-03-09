import json
import os
from typing import List, Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from global_utils.playlist_preprocessing import iter_playlists_from_dir

def build_tracks_matrix(train_dir: str) -> Tuple[csr_matrix, Dict[str, int], Dict[int, int]]:
    """
    Matriz Playlist-Track:
      - filas = playlists
      - columnas = tracks
      - valor = 1 si el track aparece en la playlist
    Devuelve: (matrix, track_to_idx, pid_to_row)
    """
    track_to_idx: Dict[str, int] = {}
    pid_to_row: Dict[int, int] = {}

    rows = []
    cols = []
    values = []

    for pl in iter_playlists_from_dir(train_dir):
        pid = pl.get("pid")
        if pid is None:
            continue

        # asigna índice de fila a cada playlist
        row = pid_to_row.get(pid)
        if row is None:
            row = len(pid_to_row)
            pid_to_row[pid] = row

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

            rows.append(row)
            cols.append(col)
            values.append(1)

    n_playlists = len(pid_to_row)
    n_tracks = len(track_to_idx)

    X = csr_matrix(
        (np.array(values, dtype=np.uint8), (np.array(rows), np.array(cols))),
        shape=(n_playlists, n_tracks),
        dtype=np.uint8
    )
    X.sum_duplicates()  # por si acaso

    return X, track_to_idx, pid_to_row

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

def build_global_popularity(train_dir: str) -> List[Tuple[str, int]]:
    X, track_to_idx, _ = build_tracks_matrix(train_dir)
    popularity_list = popularity_from_matrix(X, track_to_idx)
    return popularity_list

def recommend_for_playlist(seed: set, popularity_list: List[Tuple[str, int]], k: int = 500) -> List[str]:
    """
    Dada una playlist semilla (set de track_uri) y una lista de tracks ordenada por popularidad,
    devuelve las top-k recomendaciones (sin incluir los seeds).
    """
    recs = []
    for track_uri, _ in popularity_list:
        if track_uri not in seed:
            recs.append(track_uri)
        if len(recs) >= k:
            break
    return recs