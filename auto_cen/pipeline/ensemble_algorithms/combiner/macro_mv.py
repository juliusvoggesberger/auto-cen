"""
Implements a wrapper for the pusion Macro Majority Vote implementation.
"""

from ConfigSpace import ConfigurationSpace

from auto_cen.constants import COMBINATION, MULTICLASS, MULTILABEL, LABELS, UTILITY_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import MacroMajorityVoteCombiner


class MacroMV(BaseMethod):
    """
    Wrapper for the pusion combiner Macro Majority Vote.
    """

    def __init__(self, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.model = MacroMajorityVoteCombiner()

    def fit(self, X: list, y: list):
        # Macro Majority Vote has no training procedure
        return

    def predict(self, X: list) -> list:
        return self.model.combine(X)

    def predict_proba(self, X: list) -> list:
        pass

    def get_params(self, deep=True) -> dict:
        return {'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'MAMV',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': UTILITY_COMBINER,
                'input': (LABELS,),
                'problem': (MULTICLASS, MULTILABEL),
                'output': (LABELS,)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        return ConfigurationSpace()
