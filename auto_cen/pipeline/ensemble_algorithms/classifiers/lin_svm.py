"""
Implements a wrapper for the sklearn LSVM implementation.
"""
import warnings
import numpy as np
from scipy.special import softmax

from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from ConfigSpace import (ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter,
                         ForbiddenEqualsClause, ForbiddenAndConjunction)

from auto_cen.constants import LABELS, MULTICLASS, MIXED, CLASSIFICATION, CONTINUOUS_OUT
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod

warnings.filterwarnings('ignore',
                        category=ConvergenceWarning)  # Suppress Convergence Warnings for LinSVC


class LinearSupportVectorMachine(BaseMethod):
    """
    Wrapper for the sklearn classifier Linear Support Vector Machine.

    :param penalty: The norm used for penalizing. Can be either l1 or l2.
    :param loss: The loss function used. Can be either hinge or squared hinge.
    :param dual: Whether to solve the dual or primal optimization problem.
                 Set to auto to automatically decide.
    :param tol: Controls the tolerance for the stopping criteria.
                Value range between [1e-5, 1e-1].
    :param C: Regularization parameter.
              Value range between [0.01, 1000].
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    :param seed: Random seed.
    """

    def __init__(self, penalty: str, loss: str, tol: float, C: float, dual: str = 'auto', **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C

        self.model = LinearSVC(penalty=self.penalty, loss=self.loss, dual=self.dual, tol=self.tol,
                               C=self.C, multi_class="ovr", fit_intercept=True, intercept_scaling=1,
                               random_state=self.seed)

    def fit(self, X: np.array, y: np.array):
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
  
        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        return self.model.predict(X)

    def predict_proba(self, X: np.array) -> np.array:
        """
        The linear svm implementation of sklearn does not allow for predict_proba.
        To allow the return of probabilities compute the confidence values of the decision function
        and rescale them to pseudo probabilities using softmax.
        """
  
        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        confidence = self.model.decision_function(X)
        return softmax(confidence)

    def get_params(self, deep=True) -> dict:
        return {'penalty': self.penalty,
                'loss': self.loss,
                'tol': self.tol,
                'C': self.C,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'LSVM',
                'algorithm': CLASSIFICATION,
                'is_deterministic': False,
                'input': MIXED,
                'problem': (MULTICLASS,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        config_space = ConfigurationSpace()

        penalty = CategoricalHyperparameter("penalty", ["l1", "l2"], default_value="l2")
        loss = CategoricalHyperparameter("loss", ["hinge", "squared_hinge"],
                                         default_value="squared_hinge")

        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, log=True, default_value=1e-4)
        C = UniformFloatHyperparameter("C", 0.01, 1000, log=True, default_value=1.0)

        forbid_l1_hinge = ForbiddenAndConjunction(ForbiddenEqualsClause(penalty, "l1"),
                                                  ForbiddenEqualsClause(loss, "hinge"))
        forbid_l1_dual = ForbiddenAndConjunction(ForbiddenEqualsClause(penalty, "l1"))
        forbid_l2_dual_hinge = ForbiddenAndConjunction(ForbiddenEqualsClause(penalty, "l2"),
                                                       ForbiddenEqualsClause(loss, "hinge"))

        config_space.add_hyperparameters([C, tol, penalty, loss])

        config_space.add_forbidden_clauses([forbid_l1_hinge, forbid_l1_dual, forbid_l2_dual_hinge])

        return config_space
