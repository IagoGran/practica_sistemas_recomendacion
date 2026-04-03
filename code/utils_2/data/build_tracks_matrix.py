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
    Append the playlists of one directory to the buffers of a sparse matrix.

    Parameters
    ----------
    dataset_dir:
        Directory containing playlist JSON files.
    playlist_id_to_row:
        Mapping from playlist id to row index. It is updated in place.
    row_to_playlist_id:
        Reverse mapping from row index to playlist id. It is updated in place.
    track_to_idx:
        Mapping from track URI to column index. It is updated in place.
    rows:
        Row buffer used later to build the sparse matrix.
    cols:
        Column buffer used later to build the sparse matrix.
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

        # Each playlist contributes at most one interaction per track URI.
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
    Build the playlist-track matrix used by PureSVD.

    Parameters
    ----------
    train_dir:
        Directory containing the training playlist JSON files.
    test_dir:
        Optional directory containing the test playlist JSON files. When
        provided, train and test are appended into the same joint matrix.
    verbose:
        Whether to print progress information while building the matrix.

    Returns
    -------
    Tuple[csr_matrix, Dict[str, int], Dict[int, str], Dict[int, int], Dict[int, int]]
        Joint tuple containing:

        - the sparse playlist-track matrix
        - ``track_uri -> column`` mapping
        - ``column -> track_uri`` mapping
        - ``playlist_id -> row`` mapping
        - ``row -> playlist_id`` mapping
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
    matrix.sort_indices()

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
