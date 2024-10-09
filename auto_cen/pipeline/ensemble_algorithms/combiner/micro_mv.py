"""
Implements a wrapper for the pusion Micro Majority Vote implementation.
"""

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from auto_cen.constants import COMBINATION, MULTICLASS, MULTILABEL, LABELS, UTILITY_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import MicroMajorityVoteCombiner


class MicroMV(BaseMethod):
    """
    Wrapper for the pusion combiner Micro Majority Vote.

    :param threshold: Threshold for the majority vote. Value range [1e-2, 1].
    """

    def __init__(self, threshold, **kwargs):
        super().__init__(feature_mask=kwargs.get("feature_mask", None),
                         seed=kwargs.get("seed", None))

        self.threshold = threshold

        self.model = MicroMajorityVoteCombiner(threshold=threshold)

    def fit(self, X: list, y: list):
        # Micro Majority Vote has no training procedure
        return

    def predict(self, X: list) -> list:
        return self.model.combine(X)

    def predict_proba(self, X: list) -> list:
        pass

    def get_params(self, deep=True) -> dict:
        return {'threshold': self.threshold,
                'feature_mask': self.feature_mask,
                'seed': self.seed}

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'MIMV',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': UTILITY_COMBINER,
                'input': (LABELS,),
                'problem': (MULTICLASS, MULTILABEL),
                'output': (LABELS,)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        threshold = UniformFloatHyperparameter('threshold', lower=1e-2, upper=1.0, default_value=0.5)

        c_space.add_hyperparameters([threshold])

        return c_space
