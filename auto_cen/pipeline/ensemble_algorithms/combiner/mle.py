"""
Implements a wrapper for the pusion Maximum Likelihood implementation.
"""

from ConfigSpace import ConfigurationSpace

from auto_cen.constants import COMBINATION, MULTICLASS, MULTILABEL, LABELS, CONTINUOUS_OUT, \
    TRAINABLE_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import MaximumLikelihoodCombiner


class MLE(BaseMethod):
    """
    Wrapper for the pusion combiner Maximum Likelihood Estimator.
    """

    def __init__(self, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.model = MaximumLikelihoodCombiner()

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
        return {'name': 'MLE',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': TRAINABLE_COMBINER,
                'input': (LABELS, CONTINUOUS_OUT),
                'problem': (MULTICLASS, MULTILABEL),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        return ConfigurationSpace()
