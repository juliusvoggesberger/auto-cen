"""
Implements a wrapper for the pusion Naive Bayes implementation.
"""

from ConfigSpace import ConfigurationSpace

from auto_cen.constants import COMBINATION, MULTICLASS, LABELS, CONTINUOUS_OUT, EVIDENCE_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import NaiveBayesCombiner


class NaiveBayes(BaseMethod):
    """
    Wrapper for the pusion combiner Naive Bayes.
    """

    def __init__(self, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])
        self.model = NaiveBayesCombiner()

    def fit(self, X: list, y: list):
        self.model.train(X, y)

    def predict(self, X: list) -> list:
        return self.model.combine(X)

    def predict_proba(self, X: list) -> list:
        pass

    def get_params(self, deep=True) -> dict:
        return {'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'NB',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': EVIDENCE_COMBINER,
                'input': (LABELS, CONTINUOUS_OUT),
                'problem': (MULTICLASS,),
                'output': (LABELS,)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        return ConfigurationSpace()
