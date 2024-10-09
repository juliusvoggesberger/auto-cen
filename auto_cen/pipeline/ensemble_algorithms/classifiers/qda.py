"""
Implements a wrapper for the sklearn QDA implementation.
"""
import warnings

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from auto_cen.constants import MULTICLASS, LABELS, CONTINUOUS_OUT, MIXED, CLASSIFICATION
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod

# Suppress collinearity warning, as it is irrelevant to us
warnings.filterwarnings('ignore', category=UserWarning)

class QDA(BaseMethod):
    """
        Wrapper for the sklearn classifier Quadratic Discriminant Analysis.

        :param reg_param: Regularization parameter. Value range [1e-5, 1.0].
        :param feature_mask: A list of feature indices. Used to select the features for the model.
                            Needed if Random Subspace Method is used.
        """

    def __init__(self, reg_param: float, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.reg_param = reg_param

        self.model = QuadraticDiscriminantAnalysis(reg_param=self.reg_param)

    def fit(self, X: np.array, y: np.array):
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
  
        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        return self.model.predict(X)

    def predict_proba(self, X: np.array) -> np.array:
  
        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        return self.model.predict_proba(X)

    def get_params(self, deep=True) -> dict:
        return {'reg_param': self.reg_param,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'QDA',
                'algorithm': CLASSIFICATION,
                'is_deterministic': True,
                'input': MIXED,
                'problem': (MULTICLASS,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        reg_param = UniformFloatHyperparameter('reg_param', lower=0.0, upper=1.0, default_value=0.0)

        c_space.add_hyperparameter(reg_param)

        return c_space
