"""
Implements a wrapper for the sklearn LDA implementation.
"""
from typing import Union

import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, \
    InCondition, EqualsCondition, UnParametrizedHyperparameter, Constant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from auto_cen.constants import MULTICLASS, LABELS, CONTINUOUS_OUT, MIXED, CLASSIFICATION
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod



class LDA(BaseMethod):
    """
    Wrapper for the sklearn classifier Linear Discriminant Analysis.

    :param solver: Solver used by LDA.
                   Either singular value decomposition (svd), least squares (lsqr) or
                   eigenvalue decomposition (eigen).
    :param tol: Absolute threshold for some data value to be considered significant.
                Value range [1e-5, 1e-2].
                Used in solver svd. More information can be found in the sklearn documentation.
    :param shrinkage: Used for regularization for solver lsqr and eigen.
                      Set to 'auto' for automatic shrinkage using the Ledoit-Wolf lemma.
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    """

    def __init__(self, solver: str, tol: float = 1e-4, shrinkage: str = 'auto', **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.solver = solver
        if self.solver == 'svd':
            self.shrinkage = None
        else:
            self.shrinkage = shrinkage
        self.tol = tol

        self.model = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage, tol=self.tol)

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
        return {'solver': self.solver,
                'shrinkage': self.shrinkage,
                'tol': self.tol,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'LDA',
                'algorithm': CLASSIFICATION,
                'is_deterministic': True,
                'input': MIXED,
                'problem': (MULTICLASS,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        solver = CategoricalHyperparameter('solver', ['svd', 'lsqr', 'eigen'], default_value='svd')
        tol = UniformFloatHyperparameter('tol', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
        # shrinkage = UnParametrizedHyperparameter('shrinkage', 'auto')
        shrinkage = Constant('shrinkage', 'auto')

        cond_shrink_solve = InCondition(shrinkage, solver, ['lsqr', 'eigen'])
        cond_tol_solve = EqualsCondition(tol, solver, 'svd')

        c_space.add_hyperparameters([solver, tol, shrinkage])
        c_space.add_conditions([cond_shrink_solve, cond_tol_solve])

        return c_space
