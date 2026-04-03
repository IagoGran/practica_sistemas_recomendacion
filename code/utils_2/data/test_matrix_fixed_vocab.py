from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from global_utils.playlist_preprocessing import iter_playlists_from_dir


def append_test_playlists_with_fixed_vocab(
    dataset_dir: str,
    playlist_id_to_row: Dict[int, int],
    row_to_playlist_id: Dict[int, int],
    track_to_idx: Dict[str, int],
    rows: List[int],
    cols: List[int],
) -> None:
    """
    Process test playlists using only the vocabulary learned from train.
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
                continue

            rows_append(row)
            cols_append(col)


def build_test_matrix_fixed_vocab(
    test_dir: str,
    track_to_idx: Dict[str, int],
    verbose: bool = False,
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
    """
    Build the test playlist-track matrix using the train vocabulary only.
    """
    playlist_id_to_row: Dict[int, int] = {}
    row_to_playlist_id: Dict[int, int] = {}

    rows: List[int] = []
    cols: List[int] = []

    if verbose:
        print("INFO: Procesando test_dir con vocabulario fijo de train...")

    append_test_playlists_with_fixed_vocab(
        dataset_dir=test_dir,
        playlist_id_to_row=playlist_id_to_row,
        row_to_playlist_id=row_to_playlist_id,
        track_to_idx=track_to_idx,
        rows=rows,
        cols=cols,
    )

    if verbose:
        print(
            "INFO: Procesamiento de test_dir completado. "
            f"Se han procesado {len(playlist_id_to_row)} playlists de test."
        )

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

    if verbose:
        print(
            f"INFO: Matriz test construida con shape {matrix.shape} y "
            f"{matrix.nnz} elementos no nulos."
        )

    return matrix, playlist_id_to_row, row_to_playlist_id
