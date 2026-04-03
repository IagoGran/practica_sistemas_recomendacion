from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from global_utils.playlist_preprocessing import iter_playlists_from_dir


def append_playlists_from_dir_to_matrix_data(
    dataset_dir: str,
    playlist_id_to_row: Dict[int, int],
    row_to_playlist_id: Dict[int, int],
    track_to_idx: Dict[str, int],
    rows: List[int],
    cols: List[int],
) -> None:
    """
    Process playlists from a directory and update sparse-matrix buffers.
    """
    playlist_id_to_row_get = playlist_id_to_row.get
    track_to_idx_get = track_to_idx.get
    rows_append = rows.append
    cols_append = cols.append

    for playlist in iter_playlists_from_dir(dataset_dir):
        playlist_id = playlist.get("pid")
        if playlist_id is None:
            continue

        row = playlist_id_to_row_get(playlist_id)
        if row is None:
            row = len(playlist_id_to_row)
            playlist_id_to_row[playlist_id] = row
            row_to_playlist_id[row] = playlist_id

        seen_in_playlist = set()

        for track in playlist.get("tracks", []):
            track_uri = track.get("track_uri")
            if not track_uri or track_uri in seen_in_playlist:
                continue

            seen_in_playlist.add(track_uri)

            col = track_to_idx_get(track_uri)
            if col is None:
                col = len(track_to_idx)
                track_to_idx[track_uri] = col

            rows_append(row)
            cols_append(col)


def build_tracks_matrix(
    train_dir: str,
    test_dir: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str], Dict[int, int], Dict[int, int]]:
    """
    Build a playlist-track matrix from train and, optionally, test playlists.
    """
    track_to_idx: Dict[str, int] = {}
    playlist_id_to_row: Dict[int, int] = {}
    row_to_playlist_id: Dict[int, int] = {}

    rows: List[int] = []
    cols: List[int] = []

    if verbose:
        print("INFO: Procesando train_dir para construir la matriz...")

    append_playlists_from_dir_to_matrix_data(
        dataset_dir=train_dir,
        playlist_id_to_row=playlist_id_to_row,
        row_to_playlist_id=row_to_playlist_id,
        track_to_idx=track_to_idx,
        rows=rows,
        cols=cols,
    )

    if verbose:
        print("INFO: Train_dir procesado, matriz parcial construida.")

    if test_dir is not None:
        if verbose:
            print("INFO: Procesando test_dir para anadir playlists de test a la matriz...")

        append_playlists_from_dir_to_matrix_data(
            dataset_dir=test_dir,
            playlist_id_to_row=playlist_id_to_row,
            row_to_playlist_id=row_to_playlist_id,
            track_to_idx=track_to_idx,
            rows=rows,
            cols=cols,
        )

        if verbose:
            print("INFO: Test_dir procesado, matriz completa construida.")
    elif verbose:
        print("INFO: No se proporciono test_dir; solo se proceso train_dir.")

    n_playlists = len(playlist_id_to_row)
    n_tracks = len(track_to_idx)

    data = np.ones(len(rows), dtype=np.uint8)

    matrix = csr_matrix(
        (
            data,
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(n_playlists, n_tracks),
        dtype=np.uint8,
    )
    matrix.sum_duplicates()

    idx_to_track: Dict[int, str] = {
        idx: track_uri for track_uri, idx in track_to_idx.items()
    }

    if verbose:
        print(
            f"INFO: Matriz construida con shape {matrix.shape} y "
            f"{matrix.nnz} elementos no nulos."
        )

    return (
        matrix,
        track_to_idx,
        idx_to_track,
        playlist_id_to_row,
        row_to_playlist_id,
    )

