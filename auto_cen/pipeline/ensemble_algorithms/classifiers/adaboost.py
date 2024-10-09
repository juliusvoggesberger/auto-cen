"""
Implements a wrapper for the sklearn AdaBoost implementation.
"""

import numpy as np

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from sklearn.ensemble import AdaBoostClassifier

from auto_cen.constants import MULTICLASS, LABELS, CONTINUOUS_OUT, CLASSIFICATION, MIXED
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class AdaBoost(BaseMethod):
    """
    Wrapper for the sklearn classifier for AdaBoost.

    :param n_estimators: Number of estimators used. Value in range [2,200].
    :param learning_rate: Weight for each classifier. Value in [1e-1, 2.0].
    :param algorithm: Algorithms used for boosting. Either 'SAMME' or 'SAMME.R'.
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    :param seed: Random State.
    """

    def __init__(self, n_estimators: int, learning_rate: float, algorithm: str, **kwargs):
        # Does not use class_weights
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm

        self.model = AdaBoostClassifier(algorithm=self.algorithm, learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators, random_state=self.seed)

    def fit(self, X: np.array, y: np.array):
        return self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        return self.model.predict(X)

    def predict_proba(self, X: np.array) -> np.array:
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        return self.model.predict_proba(X)

    def get_params(self, deep=True) -> dict:
        return {'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'algorithm': self.algorithm,
                'feature_mask': self.feature_mask,
                'seed': self.seed,
                }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'AB',
                'algorithm': CLASSIFICATION,
                'is_deterministic': False,
                'input': MIXED,
                'problem': (MULTICLASS,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter('n_estimators', lower=2, upper=100, default_value=50)
        learning_rate = UniformFloatHyperparameter('learning_rate', lower=1e-1, upper=2.0, default_value=0.5)
        algorithm = CategoricalHyperparameter('algorithm', ['SAMME', 'SAMME.R'], default_value='SAMME.R')

        c_space.add_hyperparameters([n_estimators, learning_rate, algorithm])

        return c_space
