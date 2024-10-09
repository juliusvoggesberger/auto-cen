"""
Implements a wrapper for the pusion Cosine Similarity implementation.
"""

from ConfigSpace import ConfigurationSpace

from auto_cen.constants import COMBINATION, LABELS, CONTINUOUS_OUT, MULTICLASS, MULTILABEL, \
    UTILITY_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import CosineSimilarityCombiner


class CosineSimilarity(BaseMethod):
    """
    Wrapper for the pusion combiner Cosine Similarity.
    """

    def __init__(self, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])
        self.model = CosineSimilarityCombiner()

    def fit(self, X: list, y: list):
        # Cosine Similarity has no training procedure
        return

    def predict(self, X: list) -> list:
        return self.model.combine(X)

    def predict_proba(self, X: list) -> list:
        return self.model.combine(X)

    def get_params(self, deep=True) -> dict:
        return {'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'COS',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': UTILITY_COMBINER,
                'input': (LABELS, CONTINUOUS_OUT),
                'problem': (MULTICLASS, MULTILABEL),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        return ConfigurationSpace()