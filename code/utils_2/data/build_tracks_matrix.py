from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from global_utils.playlist_preprocessing import iter_playlists_from_dir


FilterStats = Dict[str, int]


def _extract_unique_track_uris(playlist: dict) -> List[str]:
    """
    Extract the effective unique track URIs contributed by one playlist.

    Parameters
    ----------
    playlist:
        Playlist dictionary read from the dataset JSON files.

    Returns
    -------
    List[str]
        Ordered list of unique valid track URIs. Duplicate tracks inside the
        same playlist are removed because the matrix stores binary interactions.
    """
    unique_track_uris: List[str] = []
    seen_in_playlist = set()

    for track in playlist.get("tracks", []):
        track_uri = track.get("track_uri")
        if not track_uri or track_uri in seen_in_playlist:
            continue

        seen_in_playlist.add(track_uri)
        unique_track_uris.append(track_uri)

    return unique_track_uris


def _passes_length_filter(
    playlist_length: int,
    min_playlist_length: Optional[int],
    max_playlist_length: Optional[int],
) -> Tuple[bool, Optional[str]]:
    """
    Check whether a playlist length is inside the accepted interval.

    Parameters
    ----------
    playlist_length:
        Effective number of unique valid tracks contributed by the playlist.
    min_playlist_length:
        Optional lower bound for accepted playlists.
    max_playlist_length:
        Optional upper bound for accepted playlists.

    Returns
    -------
    Tuple[bool, Optional[str]]
        Pair ``(keep_playlist, discard_reason)`` where ``discard_reason`` is
        ``"too_short"``, ``"too_long"`` or ``None``.
    """
    if min_playlist_length is not None and playlist_length < min_playlist_length:
        return False, "too_short"
    if max_playlist_length is not None and playlist_length > max_playlist_length:
        return False, "too_long"

    return True, None


def _build_filter_stats() -> FilterStats:
    """
    Create an empty statistics dictionary for one dataset split.

    Returns
    -------
    FilterStats
        Mutable dictionary used to accumulate preprocessing counters.
    """
    return {
        "processed": 0,
        "kept": 0,
        "discarded_too_short": 0,
        "discarded_too_long": 0,
    }


def _log_filter_stats(dataset_label: str, stats: FilterStats, verbose: bool) -> None:
    """
    Print the preprocessing counters collected for one dataset split.

    Parameters
    ----------
    dataset_label:
        Human-readable label of the processed split, such as ``train`` or
        ``test``.
    stats:
        Dictionary with the collected counters.
    verbose:
        Whether logging is enabled.
    """
    if not verbose:
        return

    print(
        f"INFO: {dataset_label} -> procesadas: {stats['processed']} | "
        f"aceptadas: {stats['kept']} | "
        f"descartadas cortas: {stats['discarded_too_short']} | "
        f"descartadas largas: {stats['discarded_too_long']}"
    )


def append_playlists_from_dir_to_matrix_data(
    dataset_dir: str,
    playlist_id_to_row: Dict[int, int],
    row_to_playlist_id: Dict[int, int],
    track_to_idx: Dict[str, int],
    rows: List[int],
    cols: List[int],
    min_playlist_length: Optional[int] = None,
    max_playlist_length: Optional[int] = None,
) -> FilterStats:
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
    min_playlist_length:
        Optional lower bound applied to the effective unique length of each
        playlist before adding it to the matrix.
    max_playlist_length:
        Optional upper bound applied to the effective unique length of each
        playlist before adding it to the matrix.

    Returns
    -------
    FilterStats
        Statistics describing how many playlists were kept or filtered out.
    """
    playlist_id_to_row_get = playlist_id_to_row.get
    rows_append = rows.append
    cols_append = cols.append

    stats = _build_filter_stats()

    for playlist in iter_playlists_from_dir(dataset_dir):
        stats["processed"] += 1

        playlist_id = playlist.get("pid")
        if playlist_id is None:
            continue

        unique_track_uris = _extract_unique_track_uris(playlist)

        keep_playlist, discard_reason = _passes_length_filter(
            playlist_length=len(unique_track_uris),
            min_playlist_length=min_playlist_length,
            max_playlist_length=max_playlist_length,
        )
        if not keep_playlist:
            if discard_reason == "too_short":
                stats["discarded_too_short"] += 1
            elif discard_reason == "too_long":
                stats["discarded_too_long"] += 1
            continue

        row = playlist_id_to_row_get(playlist_id)
        if row is None:
            row = len(playlist_id_to_row)
            playlist_id_to_row[playlist_id] = row
            row_to_playlist_id[row] = playlist_id

        for track_uri in unique_track_uris:
            col = track_to_idx.get(track_uri)
            if col is None:
                col = len(track_to_idx)
                track_to_idx[track_uri] = col

            rows_append(row)
            cols_append(col)

        stats["kept"] += 1

    return stats


def build_tracks_matrix(
    train_dir: str,
    test_dir: Optional[str] = None,
    min_playlist_length_train: Optional[int] = None,
    max_playlist_length_train: Optional[int] = None,
    min_playlist_length_test: Optional[int] = None,
    max_playlist_length_test: Optional[int] = None,
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
    min_playlist_length_train:
        Optional lower bound applied only to train playlists.
    max_playlist_length_train:
        Optional upper bound applied only to train playlists.
    min_playlist_length_test:
        Optional lower bound applied only to test playlists.
    max_playlist_length_test:
        Optional upper bound applied only to test playlists.
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

    train_stats = append_playlists_from_dir_to_matrix_data(
        dataset_dir=train_dir,
        playlist_id_to_row=playlist_id_to_row,
        row_to_playlist_id=row_to_playlist_id,
        track_to_idx=track_to_idx,
        rows=rows,
        cols=cols,
        min_playlist_length=min_playlist_length_train,
        max_playlist_length=max_playlist_length_train,
    )
    _log_filter_stats("train", train_stats, verbose)

    if verbose:
        print("INFO: Train_dir procesado, matriz parcial construida.")

    if test_dir is not None:
        if verbose:
            print("INFO: Procesando test_dir para anadir playlists de test a la matriz...")

        test_stats = append_playlists_from_dir_to_matrix_data(
            dataset_dir=test_dir,
            playlist_id_to_row=playlist_id_to_row,
            row_to_playlist_id=row_to_playlist_id,
            track_to_idx=track_to_idx,
            rows=rows,
            cols=cols,
            min_playlist_length=min_playlist_length_test,
            max_playlist_length=max_playlist_length_test,
        )
        _log_filter_stats("test", test_stats, verbose)

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
