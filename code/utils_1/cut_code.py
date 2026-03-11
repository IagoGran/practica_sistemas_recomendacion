from typing import List, Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix

def _compute_playlist_similarity_matrix(X: csr_matrix) -> csr_matrix:
    """
    Calcula la matriz de similitud coseno entre playlists
    sin densificar la matriz completa.
    """
    gram = X.dot(X.T).tocoo()  # formato COO: filas, cols, datos
    norms = np.sqrt(X.sum(axis=1)).A1.astype(np.float32)

    rows = gram.row
    cols = gram.col
    data = gram.data.astype(np.float32)

    # denominador para cada entrada no nula
    denom = norms[rows] * norms[cols]

    # evitar divisiones por cero
    valid = denom > 0
    sim_data = np.zeros_like(data, dtype=np.float32)
    sim_data[valid] = data[valid] / denom[valid]

    similarity_matrix = csr_matrix((sim_data, (rows, cols)), shape=gram.shape, dtype=np.float32)
    similarity_matrix.eliminate_zeros()

    return similarity_matrix

def _get_similar_playlists_for_playlist(
                                        playlist_id: int,
                                        similarity_matrix: csr_matrix,
                                        playlist_id_to_row: Dict[int, int],
                                        row_to_playlist_id: Dict[int, int]
                                    ) -> List[Tuple[int, int, float]]:
    similar_playlists = []

    row = playlist_id_to_row.get(playlist_id)
    if row is None:
        return similar_playlists

    row_data = similarity_matrix.getrow(row)
    cols = row_data.indices
    sims = row_data.data

    for other_row, sim in zip(cols, sims):
        if other_row != row and sim > 0:
            other_playlist_id = row_to_playlist_id.get(other_row)
            if other_playlist_id is not None:
                similar_playlists.append((other_row, other_playlist_id, float(sim)))

    similar_playlists.sort(key=lambda x: x[2], reverse=True)
    return similar_playlists