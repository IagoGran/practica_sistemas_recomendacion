from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


@dataclass(frozen=True)
class BlockResult:
    """
    Recommendation result for a single row block.
    """

    start: int
    end: int
    top_indices: np.ndarray
    top_scores: np.ndarray


class PureSVDRecommender:
    """
    PureSVD recommender for sparse playlist-track matrices.
    """

    def __init__(
        self,
        n_factors: int = 100,
        random_state: Optional[int] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
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
        if not self.is_fitted_:
            raise RuntimeError(
                "El modelo no esta entrenado todavia. Debes llamar antes a fit()."
            )

    def _validate_input_matrix(self, matrix: csr_matrix, matrix_name: str = "X") -> None:
        if not isinstance(matrix, csr_matrix):
            raise TypeError(f"{matrix_name} debe ser una scipy.sparse.csr_matrix")
        if matrix.ndim != 2:
            raise ValueError(f"{matrix_name} debe ser bidimensional")

        n_rows, n_cols = matrix.shape
        if n_rows == 0 or n_cols == 0:
            raise ValueError(f"{matrix_name} no puede tener dimensiones vacias")

    def fit(self, matrix: csr_matrix) -> "PureSVDRecommender":
        """
        Train the model with sparse SVD.
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
        Folding-in for unseen playlists.
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
        Return user factors for the requested block.
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
        Score all items for a batch of user factors.
        """
        self._validate_fitted()

        user_factors = np.asarray(user_factors, dtype=self.dtype)

        if user_factors.ndim != 2:
            raise ValueError("user_factors debe ser una matriz 2D")
        if user_factors.shape[1] != self.effective_n_factors_:
            raise ValueError(
                "Numero de columnas de user_factors incompatible con el modelo"
            )

        scores = user_factors @ self.item_factors_.T
        return np.asarray(scores, dtype=self.dtype)

    @staticmethod
    def _filter_seen_items_inplace(scores: np.ndarray, seen_matrix: csr_matrix) -> None:
        """
        Set scores of seen items to -inf in place.
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
    def _top_k_from_score_block(
        scores: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the top-k items per row.
        """
        if scores.ndim != 2:
            raise ValueError("scores debe ser una matriz 2D")
        if top_k <= 0:
            raise ValueError("top_k debe ser mayor que 0")

        _, n_items = scores.shape
        k = min(top_k, n_items)

        top_idx_unsorted = np.argpartition(scores, -k, axis=1)[:, -k:]
        top_scores_unsorted = np.take_along_axis(scores, top_idx_unsorted, axis=1)

        order = np.argsort(top_scores_unsorted, axis=1)[:, ::-1]
        top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)
        top_scores = np.take_along_axis(top_scores_unsorted, order, axis=1)

        return top_idx.astype(np.int32, copy=False), top_scores

    def recommend_block(
        self,
        *,
        start: int,
        end: int,
        X_query_block: csr_matrix,
        top_k: int = 500,
        use_folding_in: bool = False,
        filter_seen: bool = True,
    ) -> BlockResult:
        """
        Recommend for a single block of rows.
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

        scores_block = self.score_all_items(user_factors_block)

        if filter_seen:
            self._filter_seen_items_inplace(scores_block, X_query_block)

        top_idx_block, top_scores_block = self._top_k_from_score_block(
            scores=scores_block,
            top_k=top_k,
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sequential batched recommendation fallback.
        """
        self._validate_fitted()
        self._validate_input_matrix(X_query, matrix_name="X_query")

        n_rows = X_query.shape[0]
        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que 0")

        top_indices = np.empty((n_rows, top_k), dtype=np.int32)
        top_scores = np.empty((n_rows, top_k), dtype=self.dtype)

        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            X_block = X_query[start:end]

            block_result = self.recommend_block(
                start=start,
                end=end,
                X_query_block=X_block,
                top_k=top_k,
                use_folding_in=use_folding_in,
                filter_seen=filter_seen,
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper for a single-block recommendation.
        """
        return self.recommend_top_k_in_batches(
            X_query=X_query,
            top_k=top_k,
            batch_size=X_query.shape[0],
            use_folding_in=use_folding_in,
            filter_seen=filter_seen,
        )
