"""
Implements a wrapper for the sklearn ExtraTrees implementation.
"""

import numpy as np

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from sklearn.ensemble import ExtraTreesClassifier

from auto_cen.constants import CLASSIFICATION, MULTILABEL, MULTICLASS, CONTINUOUS_OUT, LABELS, \
    MIXED
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class ExtraTrees(BaseMethod):
    """
    Wrapper for the sklearn classifier for ExtraTrees.

    :param n_estimators: Number of decision trees used in the forest. Value in range [2, 200].
    :param criterion: Split criterion used for the decision tree. Either gini or entropy.
    :param max_depth_fraction: Controls the maximum depth of the decision tree.
                               Value range between [0.0,2.0].
                               The fraction is multiplied by the n_features of the dataset to
                               compute the maximum depth.
    :param min_samples_split: Minimum number of samples required for a split.
                            Value range between [2,10].
    :param min_samples_leaf: Minimum number of samples required for a leaf node.
                             Value range between [1,10].
    :param max_features: The fraction of features that are evaluated for each split.
                         Value in range [0.1,1.0]
    :param bootstrap: If True, bootstrap samples are used for tree building.
    :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
                      Value range between [5e-8,1e-3].
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    :param seed: Random state.
    """

    def __init__(self, n_estimators: int, criterion: str, max_depth_fraction: float,
                 min_samples_split: int, min_samples_leaf: int, max_features: float,
                 bootstrap: bool, ccp_alpha: float, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.n_estimators = n_estimators
        self.criterion = criterion

        self.max_depth_fraction = max_depth_fraction
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.max_features = max_features
        self.bootstrap = bootstrap
        self.ccp_alpha = ccp_alpha

        self.max_depth = 0
        self.model = None

    def fit(self, X: np.array, y: np.array):
        n_features = X.shape[1]
        # Depth must be min. 1.
        self.max_depth = max(1, int(self.max_depth_fraction * n_features))

        self.model = ExtraTreesClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                          max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf,
                                          min_samples_split=self.min_samples_split,
                                          max_features=self.max_features,
                                          bootstrap=self.bootstrap,
                                          ccp_alpha=self.ccp_alpha,
                                          random_state=self.seed)

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
        return {'n_estimators': self.n_estimators,
                'criterion': self.criterion,
                'max_depth_fraction': self.max_depth_fraction,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'ccp_alpha': self.ccp_alpha,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'XT',
                'algorithm': CLASSIFICATION,
                'is_deterministic': False,
                'input': MIXED,
                'problem': (MULTICLASS, MULTILABEL),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter('n_estimators', lower=2, upper=200,
                                                    default_value=200)
        criterion = CategoricalHyperparameter('criterion', ['gini', 'entropy'],
                                              default_value='gini')

        max_depth_fraction = UniformFloatHyperparameter('max_depth_fraction', lower=0.0, upper=2.0,
                                                        default_value=0.5)
        min_samples_split = UniformIntegerHyperparameter('min_samples_split', lower=2, upper=10,
                                                         default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', lower=1, upper=10,
                                                        default_value=1)

        max_features = UniformFloatHyperparameter('max_features', lower=0.1, upper=1.0,
                                                  default_value=0.5)
        bootstrap = CategoricalHyperparameter('bootstrap', [True, False], default_value=False)
        ccp_alpha = UniformFloatHyperparameter('ccp_alpha', lower=0, upper=1e-3, default_value=0)

        c_space.add_hyperparameters(
            [n_estimators, criterion, max_depth_fraction, min_samples_split, min_samples_leaf,
             max_features, bootstrap, ccp_alpha])

        return c_space
