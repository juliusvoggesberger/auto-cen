"""
Implements a wrapper for the pusion BKS implementation.
"""

import numpy as np

from ConfigSpace import ConfigurationSpace

from auto_cen.constants import LABELS, MULTICLASS, COMBINATION, TRAINABLE_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import BehaviourKnowledgeSpaceCombiner


class BKS(BaseMethod):
    """
    Wrapper for the pusion combiner Behaviour Knowledge Space.
    """

    def __init__(self, **kwargs):
        super().__init__(feature_mask=kwargs.get("feature_mask", None),
                         seed=kwargs.get("seed", None))
        self.model = BehaviourKnowledgeSpaceCombiner()

    def fit(self, X: np.array, y: np.array):
        self.model.train(X, y)

    def predict(self, X: np.array) -> np.array:
        return self.model.combine(X)

    def predict_proba(self, X: np.array) -> np.array:
        raise NotImplementedError('Can not predict class probabilities!')

    def get_params(self, deep=True) -> dict:
        return {'feature_mask': self.feature_mask,
                'seed': self.seed}

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'BKS',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': TRAINABLE_COMBINER,
                'input': (LABELS,),
                'problem': (MULTICLASS,),
                'output': (LABELS,)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        return ConfigurationSpace()
