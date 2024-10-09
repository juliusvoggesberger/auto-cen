"""
A wrapper for the sklearn implementation of K-nearest-neighbors.
"""

import warnings
import numpy as np

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter
from sklearn.neighbors import KNeighborsClassifier

from auto_cen.constants import CONTINUOUS_OUT, LABELS, MULTICLASS, MIXED, MULTILABEL, \
    CLASSIFICATION
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class KNearestNeighbour(BaseMethod):
    """
    Wrapper for the sklearn classifier K Neighbors.

    :param n_neighbors: Number of neighbors used. Value range [1,50].
    :param weights: Weighting used. Either 'uniform' or 'distance'
    :param p: Power parameter of minkowski distance. Either 1 (manhattan) or 2 (euclidean).
              For arbitrary p, minkowski_distance (l_p) is used.
    :param seed: Random seed.
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    """

    def __init__(self, n_neighbors, weights, p, **kwargs):
        # Does not use seeds
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.model = None

    def fit(self, X: np.array, y: np.array):
        if X.shape[0] < self.n_neighbors:
            self.n_neighbors = X.shape[0]
            warnings.warn(f"For kNN k={self.n_neighbors} was chosen, but given data has only "
                          f"{X.shape[0]} instances. Setting k={X.shape[0]}")

        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights,
                                          p=self.p)
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
        return {'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'p': self.p,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'KNN_CLASS',
                'algorithm': CLASSIFICATION,
                'is_deterministic': True,
                'input': MIXED,
                'problem': (MULTICLASS, MULTILABEL),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:

        c_space = ConfigurationSpace()
        n_neighbors = UniformIntegerHyperparameter('n_neighbors', lower=1, upper=50,
                                                   default_value=5)
        weights = CategoricalHyperparameter('weights', ['uniform', 'distance'],
                                            default_value='uniform')
        p = UniformIntegerHyperparameter('p', lower=1, upper=2, default_value=2)

        c_space.add_hyperparameters([n_neighbors, weights, p])

        return c_space
