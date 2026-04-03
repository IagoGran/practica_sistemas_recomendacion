from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


@dataclass(frozen=True)
class BlockResult:
    """
    Store the recommendation output produced for a contiguous block of rows.

    Attributes
    ----------
    start:
        Global inclusive row offset of the block inside the queried matrix.
    end:
        Global exclusive row offset of the block inside the queried matrix.
    top_indices:
        Matrix of shape ``(end - start, top_k)`` with the selected item indices.
    top_scores:
        Matrix of shape ``(end - start, top_k)`` with the scores aligned with
        ``top_indices``.
    """

    start: int
    end: int
    top_indices: np.ndarray
    top_scores: np.ndarray


class PureSVDRecommender:
    """
    Implement a PureSVD recommender for sparse playlist-track interaction data.

    The model keeps the matrices required to support both execution modes used
    in this project:

    - direct recommendation over users seen during training
    - folding-in projection for unseen playlists represented with the same item
      vocabulary as the training matrix

    Parameters
    ----------
    n_factors:
        Target rank used by the truncated SVD decomposition.
    random_state:
        Seed forwarded to ``scipy.sparse.linalg.svds``.
    dtype:
        Numeric precision used internally by the model. Only ``float32`` and
        ``float64`` are accepted.
    """

    def __init__(
        self,
        n_factors: int = 100,
        random_state: Optional[int] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """
        Build an empty recommender with the requested configuration.
        """
        if not isinstance(n_factors, int):
            raise TypeError("n_factors debe ser un entero")
        if n_factors <= 0:
            raise ValueError("n_factors debe ser mayor que 0")

        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype debe ser np.float32 o np.float64")

        self.n_factors = n_factors
        self.random_state = random_state
        self.dtype = dtype

        self.user_factors_: Optional[np.ndarray] = None
        self.s_: Optional[np.ndarray] = None
        self.s_inv_: Optional[np.ndarray] = None
        self.vt_: Optional[np.ndarray] = None
        self.v_: Optional[np.ndarray] = None
        self.item_factors_: Optional[np.ndarray] = None

        self.train_shape_: Optional[Tuple[int, int]] = None
        self.n_users_: Optional[int] = None
        self.n_items_: Optional[int] = None
        self.effective_n_factors_: Optional[int] = None

        self.is_fitted_: bool = False

    def _validate_fitted(self) -> None:
        """
        Ensure that the model has already been trained.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "El modelo no esta entrenado todavia. Debes llamar antes a fit()."
            )

    def _validate_input_matrix(self, matrix: csr_matrix, matrix_name: str = "X") -> None:
        """
        Validate that a sparse matrix is a non-empty CSR matrix.

        Parameters
        ----------
        matrix:
            Matrix to validate.
        matrix_name:
            Name used in validation errors.
        """
        if not isinstance(matrix, csr_matrix):
            raise TypeError(f"{matrix_name} debe ser una scipy.sparse.csr_matrix")
        if matrix.ndim != 2:
            raise ValueError(f"{matrix_name} debe ser bidimensional")

        n_rows, n_cols = matrix.shape
        if n_rows == 0 or n_cols == 0:
            raise ValueError(f"{matrix_name} no puede tener dimensiones vacias")

    def _validate_user_factors(self, user_factors: np.ndarray) -> np.ndarray:
        """
        Validate and normalize a user-factor matrix before scoring.

        Parameters
        ----------
        user_factors:
            Dense matrix containing one latent vector per playlist.

        Returns
        -------
        np.ndarray
            The validated factor matrix converted to the model dtype.
        """
        self._validate_fitted()

        user_factors = np.asarray(user_factors, dtype=self.dtype)
        if user_factors.ndim != 2:
            raise ValueError("user_factors debe ser una matriz 2D")
        if user_factors.shape[1] != self.effective_n_factors_:
            raise ValueError(
                "Numero de columnas de user_factors incompatible con el modelo"
            )

        return user_factors

    def _resolve_top_k(self, top_k: int) -> int:
        """
        Resolve the effective value of ``top_k`` against the item vocabulary size.

        Parameters
        ----------
        top_k:
            Requested recommendation cutoff.

        Returns
        -------
        int
            Effective cutoff, limited by the number of items in the model.
        """
        self._validate_fitted()

        if top_k <= 0:
            raise ValueError("top_k debe ser mayor que 0")

        return min(top_k, self.n_items_)

    def _resolve_item_block_size(self, item_block_size: Optional[int]) -> int:
        """
        Resolve how many items should be scored at once inside a block.

        Parameters
        ----------
        item_block_size:
            Maximum number of items scored in a single dense matrix product. If
            ``None``, all items are scored at once.

        Returns
        -------
        int
            Effective item-block size.
        """
        self._validate_fitted()

        if item_block_size is None:
            return self.n_items_
        if item_block_size <= 0:
            raise ValueError("item_block_size debe ser mayor que 0")

        return min(item_block_size, self.n_items_)

    def fit(self, matrix: csr_matrix) -> "PureSVDRecommender":
        """
        Train the recommender from a sparse interaction matrix.

        Parameters
        ----------
        matrix:
            Training matrix of shape ``(n_playlists, n_tracks)`` in CSR format.

        Returns
        -------
        PureSVDRecommender
            The fitted recommender instance.
        """
        self._validate_input_matrix(matrix, matrix_name="X")

        n_users, n_items = matrix.shape
        max_rank = min(n_users, n_items) - 1
        if max_rank <= 0:
            raise ValueError(
                "La matriz es demasiado pequena para aplicar svds. "
                "Se necesita min(n_rows, n_cols) > 1."
            )

        k = min(self.n_factors, max_rank)
        matrix_float = matrix.astype(self.dtype, copy=False)

        u, s, vt = svds(
            matrix_float,
            k=k,
            return_singular_vectors=True,
            random_state=self.random_state,
        )

        # ``svds`` devuelve los valores singulares sin ordenar, por eso
        # reordenamos el modelo entero antes de guardarlo.
        order = np.argsort(s)[::-1]
        u = u[:, order]
        s = s[order]
        vt = vt[order, :]

        u = np.ascontiguousarray(u, dtype=self.dtype)
        s = np.ascontiguousarray(s, dtype=self.dtype)
        vt = np.ascontiguousarray(vt, dtype=self.dtype)

        eps = np.finfo(self.dtype).eps
        s_inv = np.where(s > eps, 1.0 / s, 0.0).astype(self.dtype, copy=False)
        v = np.ascontiguousarray(vt.T, dtype=self.dtype)
        item_factors = np.ascontiguousarray(v * s[np.newaxis, :], dtype=self.dtype)

        self.user_factors_ = u
        self.s_ = s
        self.s_inv_ = s_inv
        self.vt_ = vt
        self.v_ = v
        self.item_factors_ = item_factors

        self.train_shape_ = matrix.shape
        self.n_users_ = n_users
        self.n_items_ = n_items
        self.effective_n_factors_ = k
        self.is_fitted_ = True
        return self

    def project_users(self, matrix_new: csr_matrix) -> np.ndarray:
        """
        Project unseen playlists into the latent space with folding-in.

        Parameters
        ----------
        matrix_new:
            Query matrix expressed with the same item vocabulary used during
            training.

        Returns
        -------
        np.ndarray
            Dense matrix of latent factors with shape
            ``(n_query_playlists, effective_n_factors_)``.
        """
        self._validate_fitted()
        self._validate_input_matrix(matrix_new, matrix_name="X_new")

        if matrix_new.shape[1] != self.n_items_:
            raise ValueError(
                "X_new debe tener el mismo numero de columnas que la matriz de entrenamiento"
            )

        matrix_new_float = matrix_new.astype(self.dtype, copy=False)
        user_factors = matrix_new_float @ self.v_
        user_factors = np.asarray(user_factors, dtype=self.dtype)
        user_factors *= self.s_inv_[np.newaxis, :]
        return np.ascontiguousarray(user_factors, dtype=self.dtype)

    def get_user_factors(
        self,
        X: Optional[csr_matrix] = None,
        use_folding_in: bool = False,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return user factors for a contiguous row block.

        Parameters
        ----------
        X:
            Query matrix used only when ``use_folding_in`` is ``True``.
        use_folding_in:
            Whether the factors must be projected from ``X`` instead of read
            directly from the fitted users.
        start:
            Inclusive start row for already-trained users.
        end:
            Exclusive end row for already-trained users.

        Returns
        -------
        np.ndarray
            Dense matrix of user factors for the requested rows.
        """
        self._validate_fitted()

        if use_folding_in:
            if X is None:
                raise ValueError("Si use_folding_in=True, debes proporcionar X")
            return self.project_users(X)

        if self.user_factors_ is None:
            raise RuntimeError("No hay factores de usuario entrenados")
        if X is not None:
            raise ValueError(
                "Si use_folding_in=False, no debes pasar X. "
                "Se usan directamente los factores entrenados."
            )

        start = 0 if start is None else start
        end = self.n_users_ if end is None else end

        if start < 0 or end < start or end > self.n_users_:
            raise ValueError("Rango start:end no valido")

        return self.user_factors_[start:end]

    def score_all_items(self, user_factors: np.ndarray) -> np.ndarray:
        """
        Score the complete item catalog for a block of latent users.

        Parameters
        ----------
        user_factors:
            Dense latent representation of the queried playlists.

        Returns
        -------
        np.ndarray
            Dense score matrix of shape ``(n_rows, n_items_)``.
        """
        user_factors = self._validate_user_factors(user_factors)
        scores = user_factors @ self.item_factors_.T
        return np.asarray(scores, dtype=self.dtype)

    def score_item_block(
        self,
        user_factors: np.ndarray,
        item_start: int,
        item_end: int,
    ) -> np.ndarray:
        """
        Score only a slice of the item catalog for a block of users.

        Parameters
        ----------
        user_factors:
            Dense latent representation of the queried playlists.
        item_start:
            Inclusive start index of the item slice.
        item_end:
            Exclusive end index of the item slice.

        Returns
        -------
        np.ndarray
            Dense score matrix of shape ``(n_rows, item_end - item_start)``.
        """
        user_factors = self._validate_user_factors(user_factors)

        if item_start < 0 or item_end <= item_start or item_end > self.n_items_:
            raise ValueError("Rango item_start:item_end no valido")

        item_factors_block = self.item_factors_[item_start:item_end]
        scores = user_factors @ item_factors_block.T
        return np.asarray(scores, dtype=self.dtype)

    @staticmethod
    def _filter_seen_items_inplace(scores: np.ndarray, seen_matrix: csr_matrix) -> None:
        """
        Set to ``-inf`` the scores of items already present in each playlist.

        Parameters
        ----------
        scores:
            Dense score matrix to edit in place.
        seen_matrix:
            Sparse matrix with the already-observed items per playlist. The
            matrix must have the same number of rows as ``scores``.
        """
        if not isinstance(seen_matrix, csr_matrix):
            raise TypeError("X_seen debe ser una csr_matrix")
        if scores.shape[0] != seen_matrix.shape[0]:
            raise ValueError("El numero de filas de scores y X_seen debe coincidir")

        indptr = seen_matrix.indptr
        indices = seen_matrix.indices

        for row_idx in range(seen_matrix.shape[0]):
            start = indptr[row_idx]
            end = indptr[row_idx + 1]
            seen_items = indices[start:end]
            scores[row_idx, seen_items] = -np.inf

    @staticmethod
    def _filter_seen_items_in_range_inplace(
        scores: np.ndarray,
        seen_matrix: csr_matrix,
        item_start: int,
        item_end: int,
    ) -> None:
        """
        Set to ``-inf`` only the seen items that fall inside an item slice.

        Parameters
        ----------
        scores:
            Dense score matrix for the current item slice.
        seen_matrix:
            Sparse matrix containing the items already seen by each playlist.
        item_start:
            Inclusive start index of the scored item slice.
        item_end:
            Exclusive end index of the scored item slice.
        """
        if not isinstance(seen_matrix, csr_matrix):
            raise TypeError("X_seen debe ser una csr_matrix")
        if scores.shape[0] != seen_matrix.shape[0]:
            raise ValueError("El numero de filas de scores y X_seen debe coincidir")

        indptr = seen_matrix.indptr
        indices = seen_matrix.indices

        for row_idx in range(seen_matrix.shape[0]):
            start = indptr[row_idx]
            end = indptr[row_idx + 1]
            row_seen = indices[start:end]

            # Las columnas del CSR estan ordenadas; eso permite localizar solo
            # los vistos que caen en este rango sin escanear el bloque entero.
            local_start = np.searchsorted(row_seen, item_start, side="left")
            local_end = np.searchsorted(row_seen, item_end, side="left")
            if local_end <= local_start:
                continue

            local_seen = row_seen[local_start:local_end] - item_start
            scores[row_idx, local_seen] = -np.inf

    @staticmethod
    def _top_k_from_score_block(
        scores: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the best-scoring items per row from a dense score matrix.

        Parameters
        ----------
        scores:
            Dense score matrix with shape ``(n_rows, n_candidates)``.
        top_k:
            Requested cutoff.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Pair ``(top_indices, top_scores)`` ordered from highest to lowest
            score per row.
        """
        if scores.ndim != 2:
            raise ValueError("scores debe ser una matriz 2D")
        if top_k <= 0:
            raise ValueError("top_k debe ser mayor que 0")

        _, n_items = scores.shape
        k = min(top_k, n_items)

        if k == n_items:
            order = np.argsort(scores, axis=1)[:, ::-1]
            top_idx = order
            top_scores = np.take_along_axis(scores, order, axis=1)
            return top_idx.astype(np.int32, copy=False), top_scores

        top_idx_unsorted = np.argpartition(scores, -k, axis=1)[:, -k:]
        top_scores_unsorted = np.take_along_axis(scores, top_idx_unsorted, axis=1)

        order = np.argsort(top_scores_unsorted, axis=1)[:, ::-1]
        top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)
        top_scores = np.take_along_axis(top_scores_unsorted, order, axis=1)

        return top_idx.astype(np.int32, copy=False), top_scores

    def _merge_top_k_candidates(
        self,
        best_indices: np.ndarray,
        best_scores: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge the accumulated best items with the candidates of one item slice.

        Parameters
        ----------
        best_indices:
            Accumulated best item indices per row.
        best_scores:
            Accumulated best scores per row.
        candidate_indices:
            Item indices selected inside the current item slice.
        candidate_scores:
            Scores aligned with ``candidate_indices``.
        top_k:
            Final cutoff to keep after merging both candidate sets.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Updated pair ``(best_indices, best_scores)``.
        """
        combined_indices = np.concatenate((best_indices, candidate_indices), axis=1)
        combined_scores = np.concatenate((best_scores, candidate_scores), axis=1)

        # El merge solo necesita decidir entre el top-k acumulado y el top-k del
        # bloque actual; no hace falta reordenar los millones de items otra vez.
        merged_pos, merged_scores = self._top_k_from_score_block(combined_scores, top_k)
        merged_indices = np.take_along_axis(combined_indices, merged_pos, axis=1)

        return merged_indices.astype(np.int32, copy=False), merged_scores

    def _recommend_from_item_blocks(
        self,
        user_factors_block: np.ndarray,
        X_query_block: csr_matrix,
        top_k: int,
        filter_seen: bool,
        item_block_size: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute an exact top-k by scanning the item catalog in manageable slices.

        Parameters
        ----------
        user_factors_block:
            Dense latent representation of the queried playlists.
        X_query_block:
            Sparse query block used to filter seen tracks.
        top_k:
            Requested cutoff.
        filter_seen:
            Whether already-observed tracks must be excluded.
        item_block_size:
            Number of items scored at once. ``None`` means score the whole
            catalog in a single dense product.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Final exact top-k indices and scores for the block.
        """
        effective_top_k = self._resolve_top_k(top_k)
        effective_item_block_size = self._resolve_item_block_size(item_block_size)

        if not X_query_block.has_sorted_indices:
            X_query_block = X_query_block.sorted_indices()

        n_rows = X_query_block.shape[0]
        best_indices = np.full((n_rows, effective_top_k), -1, dtype=np.int32)
        best_scores = np.full(
            (n_rows, effective_top_k),
            fill_value=-np.inf,
            dtype=self.dtype,
        )

        for item_start in range(0, self.n_items_, effective_item_block_size):
            item_end = min(item_start + effective_item_block_size, self.n_items_)
            score_slice = self.score_item_block(
                user_factors=user_factors_block,
                item_start=item_start,
                item_end=item_end,
            )

            if filter_seen:
                self._filter_seen_items_in_range_inplace(
                    scores=score_slice,
                    seen_matrix=X_query_block,
                    item_start=item_start,
                    item_end=item_end,
                )

            local_top_idx, local_top_scores = self._top_k_from_score_block(
                scores=score_slice,
                top_k=effective_top_k,
            )
            local_top_idx = local_top_idx.astype(np.int32, copy=False) + item_start

            best_indices, best_scores = self._merge_top_k_candidates(
                best_indices=best_indices,
                best_scores=best_scores,
                candidate_indices=local_top_idx,
                candidate_scores=local_top_scores,
                top_k=effective_top_k,
            )

        return best_indices, best_scores

    def recommend_block(
        self,
        *,
        start: int,
        end: int,
        X_query_block: csr_matrix,
        top_k: int = 500,
        use_folding_in: bool = False,
        filter_seen: bool = True,
        item_block_size: Optional[int] = 50_000,
    ) -> BlockResult:
        """
        Recommend the best items for one contiguous block of playlists.

        Parameters
        ----------
        start:
            Inclusive global row offset of the block.
        end:
            Exclusive global row offset of the block.
        X_query_block:
            Sparse query matrix for the block.
        top_k:
            Requested recommendation cutoff.
        use_folding_in:
            Whether the user factors must be projected from ``X_query_block``.
        filter_seen:
            Whether already-seen tracks must be removed from the ranking.
        item_block_size:
            Number of items scored at once. Smaller values reduce peak memory at
            the cost of doing more matrix products.

        Returns
        -------
        BlockResult
            Exact top-k indices and scores for the block.
        """
        self._validate_fitted()
        self._validate_input_matrix(X_query_block, matrix_name="X_query_block")

        n_block_rows = end - start
        if n_block_rows <= 0:
            raise ValueError("El bloque debe tener al menos una fila")
        if X_query_block.shape[0] != n_block_rows:
            raise ValueError("X_query_block.shape[0] debe coincidir con end - start")

        if use_folding_in:
            user_factors_block = self.project_users(X_query_block)
        else:
            user_factors_block = self.get_user_factors(
                X=None,
                use_folding_in=False,
                start=start,
                end=end,
            )

        top_idx_block, top_scores_block = self._recommend_from_item_blocks(
            user_factors_block=user_factors_block,
            X_query_block=X_query_block,
            top_k=top_k,
            filter_seen=filter_seen,
            item_block_size=item_block_size,
        )

        return BlockResult(
            start=start,
            end=end,
            top_indices=top_idx_block,
            top_scores=top_scores_block,
        )

    def recommend_top_k_in_batches(
        self,
        X_query: csr_matrix,
        top_k: int = 500,
        batch_size: int = 1000,
        use_folding_in: bool = False,
        filter_seen: bool = True,
        item_block_size: Optional[int] = 50_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recommend in sequential playlist batches.

        Parameters
        ----------
        X_query:
            Sparse query matrix.
        top_k:
            Requested recommendation cutoff.
        batch_size:
            Number of playlists processed per outer batch.
        use_folding_in:
            Whether the playlists must be projected with folding-in.
        filter_seen:
            Whether already-seen tracks must be removed from the ranking.
        item_block_size:
            Number of items scored at once inside each batch.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Matrices containing the final item indices and scores for every
            queried playlist.
        """
        self._validate_fitted()
        self._validate_input_matrix(X_query, matrix_name="X_query")

        n_rows = X_query.shape[0]
        effective_top_k = self._resolve_top_k(top_k)
        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que 0")

        top_indices = np.empty((n_rows, effective_top_k), dtype=np.int32)
        top_scores = np.empty((n_rows, effective_top_k), dtype=self.dtype)

        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            X_block = X_query[start:end]

            block_result = self.recommend_block(
                start=start,
                end=end,
                X_query_block=X_block,
                top_k=effective_top_k,
                use_folding_in=use_folding_in,
                filter_seen=filter_seen,
                item_block_size=item_block_size,
            )

            top_indices[start:end] = block_result.top_indices
            top_scores[start:end] = block_result.top_scores

        return top_indices, top_scores

    def recommend_top_k(
        self,
        X_query: csr_matrix,
        top_k: int = 500,
        use_folding_in: bool = False,
        filter_seen: bool = True,
        item_block_size: Optional[int] = 50_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recommend for the full query matrix in a single outer batch.

        Parameters
        ----------
        X_query:
            Sparse query matrix.
        top_k:
            Requested recommendation cutoff.
        use_folding_in:
            Whether the playlists must be projected with folding-in.
        filter_seen:
            Whether already-seen tracks must be removed from the ranking.
        item_block_size:
            Number of items scored at once inside the recommendation step.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Matrices containing the final item indices and scores for every
            queried playlist.
        """
        return self.recommend_top_k_in_batches(
            X_query=X_query,
            top_k=top_k,
            batch_size=X_query.shape[0],
            use_folding_in=use_folding_in,
            filter_seen=filter_seen,
            item_block_size=item_block_size,
        )
