"""
Implements a wrapper for the sklearn Gradient Boosting implementation.
"""

import numpy as np

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UniformFloatHyperparameter
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from auto_cen.constants import CLASSIFICATION, MULTICLASS, CONTINUOUS_OUT, LABELS, MIXED
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class GradientBoosting(BaseMethod):
    """
    Wrapper for the sklearn classifier for GradientBoosting.

    :param learning_rate: Reduction of contribution of each tree. Value in range [1e-2, 0.5].
    :param min_samples_leaf: Minimum number of samples required for a leaf node.
                             Value range between [1,10].
    :param l2_regularization: L2 regularization parameter. Value in range [1e-5, 1].

    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    :param seed: Random State.
    """

    def __init__(self, learning_rate: float, min_samples_leaf: int, l2_regularization: float,
                 **kwargs):
        # Does not use class weights
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.loss = 'log_loss'
        self.learning_rate = learning_rate

        self.min_samples_leaf = min_samples_leaf

        self.l2_regularization = l2_regularization

        self.model = HistGradientBoostingClassifier(loss=self.loss,
                                                    learning_rate=self.learning_rate,
                                                    min_samples_leaf=self.min_samples_leaf,
                                                    max_depth=None,
                                                    max_bins=255,
                                                    l2_regularization=self.l2_regularization,
                                                    early_stopping=True,
                                                    tol=1e-7,
                                                    scoring="loss",
                                                    n_iter_no_change=10,
                                                    validation_fraction=None,
                                                    random_state=self.seed)

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
        return {'learning_rate': self.learning_rate,
                'min_samples_leaf': self.min_samples_leaf,
                'l2_regularization': self.l2_regularization,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'GB',
                'algorithm': CLASSIFICATION,
                'is_deterministic': True,
                'input': MIXED,
                'problem': (MULTICLASS,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        learning_rate = UniformFloatHyperparameter('learning_rate', lower=1e-2, upper=0.5, log=True,
                                                   default_value=0.1)

        min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=200,
                                                        default_value=20)
        # max_depth = UniformIntegerHyperparameter('max_depth', lower=1, upper=10, default_value=3)

        l2_regularization = UniformFloatHyperparameter('l2_regularization', lower=1e-10, upper=1,
                                                       log=True, default_value=1e-10)

        c_space.add_hyperparameters([min_samples_leaf, learning_rate, l2_regularization])

        return c_space
