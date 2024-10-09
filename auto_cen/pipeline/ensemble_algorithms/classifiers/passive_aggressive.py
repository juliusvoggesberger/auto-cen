"""
Implements a wrapper for the sklearn Passive Aggressive implementation.
"""

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter
from scipy.special import softmax
from sklearn.linear_model import PassiveAggressiveClassifier

from auto_cen.constants import MULTICLASS, LABELS, MIXED, CLASSIFICATION, CONTINUOUS_OUT
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class PassiveAggressive(BaseMethod):
    """
    Wrapper for the sklearn classifier Passive Aggressive.

    :param C: Regularization parameter. Value range [1e-2, 10].
    :param tol: Stopping criterion. Value range [1e-5, 1e-1].
    :param loss: The loss function. Either hinge or squared hinge.
    :param average: If True, compute averaged SGD weight. Can be either True or False.
    :param early_stopping: If True, use early stopping to terminate training when validation
                          Always set to True.
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                            Needed if Random Subspace Method is used.
    :param seed: Random seed.
    """

    def __init__(self, C: float, tol: float, loss: str, average: bool, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.C = C
        self.tol = tol
        self.loss = loss
        self.average = average

        self.model = PassiveAggressiveClassifier(C=self.C, tol=self.tol, loss=self.loss,
                                                 average=self.average, early_stopping=True,
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
        The passive aggressive implementation of sklearn does not allow for predict_proba.
        To allow the return of probabilities compute the confidence values of the decision function
        and rescale them to pseudo probabilities using softmax.
        """
  
        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        confidence = self.model.decision_function(X)
        return softmax(confidence)

    def get_params(self, deep=True) -> dict:
        return {'C': self.C,
                'tol': self.tol,
                'loss': self.loss,
                'average': self.average,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'PA',
                'algorithm': CLASSIFICATION,
                'is_deterministic': False,
                'input': MIXED,
                'problem': (MULTICLASS,),
                'output': (LABELS,CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        C = UniformFloatHyperparameter('C', lower=1e-2, upper=10, log=True, default_value=1)
        tol = UniformFloatHyperparameter('tol', lower=1e-5, upper=1e-1, log=True, default_value=1e-3)

        loss = CategoricalHyperparameter('loss', ['hinge', 'squared_hinge'], default_value='hinge')
        average = CategoricalHyperparameter('average', [True, False], default_value=False)

        c_space.add_hyperparameters([C, tol, loss, average])

        return c_space
