"""
Implements a wrapper for the pusion Borda Count implementation.
"""

import numpy as np
from ConfigSpace import ConfigurationSpace

from auto_cen.constants import COMBINATION, CONTINUOUS_OUT, MULTICLASS, LABELS, UTILITY_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import BordaCountCombiner


class BordaCount(BaseMethod):
    """
    Wrapper for the pusion combiner Borda Count.
    """

    def __init__(self, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])
        self.model = BordaCountCombiner()

    def fit(self, X: np.array, y: np.array):
        # Borda Count has no training procedure
        return

    def predict(self, X: np.array) -> np.array:
        return self.model.combine(X)

    def predict_proba(self, X: np.array) -> np.array:
        raise NotImplementedError('Can not predict class probabilities!')

    def get_params(self, deep=True) -> dict:
        return {'feature_mask': self.feature_mask,
                'seed': self.seed}

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'BC',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': UTILITY_COMBINER,
                'input': (CONTINUOUS_OUT,),
                'problem': (MULTICLASS,),
                'output': (LABELS,)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        return ConfigurationSpace()
