"""
Implements a wrapper for the sklearn Gaussian Naive Bayes implementation.
"""
import numpy as np

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from sklearn.naive_bayes import GaussianNB

from auto_cen.constants import MIXED, MULTICLASS, LABELS, CONTINUOUS_OUT, CLASSIFICATION
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class GaussianNaiveBayes(BaseMethod):
    """
    Wrapper for the sklearn classifier Gaussian Naive Bayes.
    The classifier has no hyperparameters.
    :param var_smoothing: Portion of the largest variance of all features that is added to variances for calculation
                            stability.
                            Value range between [1e-11,1e-7].
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.

    """

    def __init__(self, var_smoothing: float = 1e-9, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])
        self.var_smoothing = var_smoothing

        self.model = GaussianNB(var_smoothing=self.var_smoothing)

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
        return {'var_smoothing': self.var_smoothing,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'GNB',
                'algorithm': CLASSIFICATION,
                'is_deterministic': True,
                'input': MIXED,
                'problem': (MULTICLASS,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        var_smoothing = UniformFloatHyperparameter("var_smoothing", 1e-11, 1e-7, default_value=1e-9)

        c_space.add_hyperparameters([var_smoothing])

        return c_space
