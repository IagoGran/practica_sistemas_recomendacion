"""
Expose the PureSVD model classes used by the runtime layer.
"""

from utils_2.model.pure_svd_recommender import BlockResult, PureSVDRecommender

__all__ = [
    "BlockResult",
    "PureSVDRecommender",
]
