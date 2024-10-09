"""
Implements a wrapper for the pusion knn implementation.
"""

import warnings

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter

from auto_cen.constants import COMBINATION, MULTILABEL, MULTICLASS, LABELS, CONTINUOUS_OUT, \
    TRAINABLE_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import KNNCombiner


class KNearestNeighbor(BaseMethod):
    """
    Wrapper for the pusion combiner KNNCombiner.

    :param n_neighbors: Number of neighbors used. Value range [1,50].
    :param weights: Weighting used. Either 'uniform' or 'distance'
    :param p: Power parameter of minkowski distance. Either 1 (manhattan) or 2 (euclidean).
              For arbitrary p, minkowski_distance (l_p) is used.
    """

    def __init__(self, n_neighbors: int, weights: str, p: int, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

        self.model = None

    def fit(self, X: np.array, y: np.array):
        if X.shape[1] < self.n_neighbors:
            self.n_neighbors = X.shape[0]
            warnings.warn(f"For kNN k={self.n_neighbors} was chosen, but given data has only "
                          f"{X.shape[0]} instances. Setting k={X.shape[0]}")

        self.model = KNNCombiner(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)

        self.model.train(X, y)

    def predict(self, X: list) -> list:
        return self.model.combine(X)

    def predict_proba(self, X: list) -> list:
        return self.model.combine(X, True)

    def get_params(self, deep=True) -> dict:
        return {'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'p': self.p,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'KNN_COMB',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': TRAINABLE_COMBINER,
                'input': (LABELS, CONTINUOUS_OUT),
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
