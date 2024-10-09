"""
Implements a wrapper for the sklearn SVM implementation.
"""

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter, EqualsCondition, InCondition
from scipy.special import softmax
from sklearn.svm import SVC

from auto_cen.constants import LABELS, MULTICLASS, MIXED, CLASSIFICATION, CONTINUOUS_OUT
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class SupportVectorMachine(BaseMethod):
    """
        Wrapper for the sklearn classifier Support Vector Machine.

        :param C: Regularization parameter. Value range between [0.01, 1000].
        :param kernel: Kernel to be used. Either sigmoid, poly or rbf. For linear use Linear SVM.
        :param gamma: Kernel coefficient. Value range [1e-03, 1]
        :param degree: Degree for the polynomial kernel. Value range [2,5].
        :param coef0: Kernel function term for poly and sigmoid. Value range [0.0, 2.0].
        :param feature_mask: A list of feature indices. Used to select the features for the model.
                            Needed if Random Subspace Method is used.
        :param tol: Tolerance for stopping criterion. Value range [1e-5, 1e-1].
        :param seed: Random seed.
        """

    def __init__(self, C: float, kernel: str, gamma: float, degree: int = 2, coef0: float = 0.0,
                 tol: float = 1e-3, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol

        self.model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree,
                         coef0=self.coef0,
                         probability=False, random_state=self.seed, cache_size=256, max_iter=-1,
                         tol=self.tol)

    def fit(self, X: np.array, y: np.array):
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:

        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        return self.model.predict(X)

    def predict_proba(self, X: np.array) -> np.array:
        """
        The svm implementation of sklearn does not allow for predict_proba without using CV.
        To allow the return of probabilities compute the confidence values of the decision function
        and rescale them to pseudo probabilities using softmax.
        """

        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        confidence = self.model.decision_function(X)
        return softmax(confidence)

    def get_params(self, deep=True) -> dict:

        params = {'C': self.C,
                  'kernel': self.kernel,
                  'gamma': self.gamma,
                  'tol': self.tol,
                  'feature_mask': self.feature_mask,
                  'seed': self.seed, }

        if self.kernel == 'poly':
            params['degree'] = self.degree
            params['coef0'] = self.coef0

        if self.kernel == 'sigmoid':
            params['coef0'] = self.coef0

        return params

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'SVM',
                'algorithm': CLASSIFICATION,
                'is_deterministic': False,
                'input': MIXED,
                'problem': (MULTICLASS,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        C = UniformFloatHyperparameter('C', lower=0.01, upper=1000, log=True, default_value=1.0)
        kernel = CategoricalHyperparameter('kernel', ['poly', 'rbf', 'sigmoid'],
                                           default_value='rbf')
        gamma = UniformFloatHyperparameter('gamma', lower=1e-03, upper=1, log=True,
                                           default_value=0.1)

        tol = UniformFloatHyperparameter('tol', lower=1e-5, upper=1e-1, log=True,
                                         default_value=1e-3)

        degree = UniformIntegerHyperparameter('degree', lower=2, upper=5, default_value=3)
        coef0 = UniformFloatHyperparameter('coef0', lower=0.0, upper=2.0, default_value=0.0)

        cond_poly_degree = EqualsCondition(degree, kernel, 'poly')
        cond_coef0 = InCondition(coef0, kernel, ['poly', 'sigmoid'])

        c_space.add_hyperparameters([C, gamma, tol, kernel, degree, coef0])

        c_space.add_conditions([cond_poly_degree, cond_coef0])

        return c_space
