"""
Implements a wrapper for the sklearn DecisionTree implementation.
"""

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from sklearn.tree import DecisionTreeClassifier

from auto_cen.constants import LABELS, CONTINUOUS_OUT, MULTICLASS, MULTILABEL, CLASSIFICATION, \
    MIXED
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class DecisionTree(BaseMethod):
    """
    Wrapper for the sklearn classifier for Decision Tree.

    :param criterion: Split criterion used for the decision tree. Either gini or entropy.
    :param max_depth_fraction: Controls the maximum depth of the decision tree.
                               Value range between [0.0,2.0].
                               The fraction is multiplied by the n_features of the dataset to
                               compute the maximum depth.
    :param min_samples_split: Minimum number of samples required for a split.
                              Value range between [2,10].
    :param min_samples_leaf: Minimum number of samples required for a leaf node.
                             Value range between [1,10].
    :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
                       Value range between [5e-8,1e-3].
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    :param seed: Random seed.
    """

    def __init__(self, criterion: str, max_depth_fraction: float, min_samples_split: int,
                 min_samples_leaf: int, ccp_alpha: float, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.criterion = criterion
        self.max_depth_fraction = max_depth_fraction
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.max_depth = 0
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.array):
        n_features = X.shape[1]
        # Depth must be min. 1.
        self.max_depth = max(1, int(self.max_depth_fraction * n_features))
        self.model = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split,
                                            min_samples_leaf=self.min_samples_leaf,
                                            ccp_alpha=self.ccp_alpha, random_state=self.seed)
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
        return {'criterion': self.criterion,
                'max_depth_fraction': self.max_depth_fraction,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'ccp_alpha': self.ccp_alpha,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'DTREE',
                'algorithm': CLASSIFICATION,
                'is_deterministic': False,
                'input': MIXED,
                'problem': (MULTICLASS, MULTILABEL),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        criterion = CategoricalHyperparameter('criterion', ['gini', 'entropy'], default_value='gini')

        max_depth_fraction = UniformFloatHyperparameter('max_depth_fraction', lower=0, upper=2, default_value=0.5)
        min_samples_split = UniformIntegerHyperparameter('min_samples_split', lower=2, upper=10, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=10, default_value=1)
        ccp_alpha = UniformFloatHyperparameter('ccp_alpha', lower=0, upper=1e-3, default_value=0)

        c_space.add_hyperparameters(
            [criterion, max_depth_fraction, min_samples_split, min_samples_leaf, ccp_alpha])

        return c_space
