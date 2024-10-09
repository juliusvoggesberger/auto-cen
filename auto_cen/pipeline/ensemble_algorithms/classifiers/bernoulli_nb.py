"""
Implements a wrapper for the sklearn Bernoulli Naive Bayes implementation.
"""

import numpy as np

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter
from sklearn.naive_bayes import BernoulliNB

from auto_cen.constants import CATEGORICAL, MULTICLASS, LABELS, CONTINUOUS_OUT, CLASSIFICATION
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class BernoulliNaiveBayes(BaseMethod):
    """
    Wrapper for the sklearn classifier for Bernoulli Naive Bayes.

    :param alpha: Additive smoothing parameter.
                  Alpha value is between [0.01, 10]. Range is taken as in auto-sklearn.
    :param fit_prior: True, if class priors should be computed. Else uniform priors are used.
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    """

    def __init__(self, alpha: float, fit_prior: bool, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])
        self.alpha = alpha
        self.fit_prior = fit_prior

        self.model = BernoulliNB(alpha=self.alpha, fit_prior=self.fit_prior)

    def fit(self, X: np.array, y: np.array):
        return self.model.fit(X, y)

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
        return {'alpha': self.alpha,
                'fit_prior': self.fit_prior,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'BNB',
                'algorithm': CLASSIFICATION,
                'is_deterministic': True,
                'input': CATEGORICAL,
                'problem': (MULTICLASS,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        alpha = UniformFloatHyperparameter('alpha', lower=1e-2, upper=10, log=True, default_value=1.0)
        fit_prior = CategoricalHyperparameter('fit_prior', [True, False], default_value=True)

        c_space.add_hyperparameters([alpha, fit_prior])

        return c_space
