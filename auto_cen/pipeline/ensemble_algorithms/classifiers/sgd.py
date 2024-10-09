"""
Implements a wrapper for the sklearn SGD implementation.
"""
import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, InCondition
from scipy.special import softmax
from sklearn.linear_model import SGDClassifier

from auto_cen.constants import MULTICLASS, MIXED, LABELS, CLASSIFICATION, CONTINUOUS_OUT
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class SGD(BaseMethod):
    """
    Wrapper for the sklearn classifier SGD classifier.

    :param loss: The loss function. Can have the following values:
                 hinge : Gives linear SVM.
                 log: Gives logistic regression.
                 perceptron: Loss used by perceptron classifier.
                 Further losses: modified_huber, squared_hinge.
    :param penalty: Penalty term. Can be either l1, l2 or elastic net.
    :param alpha: Multiplication term for the regularization term. Value range [1e-6, 1e-1].
    :param l1_ratio: Mixing parameter for elastic net. Value range [1e-4, 1].
    :param tol: Stopping criterion. Value range [1e-5, 1e-1].
    :param epsilon: Used for huber loss. Value range [1e-4, 1e-1].
    :param learning_rate: Learning rate. Can be 'constant', 'optimal', 'invscaling' or 'adaptive'.
    :param eta0: Initial learning rate. Value range [1e-5, 1e-1].
    :param power_t: Exponent for invscaling. Value range [1e-5, 1.0].
    :param average: If True, compute averaged SGD weight. Can be either True or False.
    :param fit_intercept: If True, the intercept should be estimated.
    :param early_stopping: If True, use early stopping to terminate training when validation
                            Always set to True.
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                            Needed if Random Subspace Method is used.
    :param seed: Random seed.
    """

    # For loss=log -> Logistic Regression
    # For loss=perceptron -> Perceptron

    def __init__(self, loss: str, penalty: str, alpha: float, tol: float,
                 epsilon: float, learning_rate: str, power_t, average: bool,
                 eta0: float = 0.0, l1_ratio: float = 0.15, fit_intercept: bool = True, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.loss = loss
        self.penalty = penalty

        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.fit_intercept = fit_intercept
        self.tol = tol
        self.epsilon = epsilon

        self.learning_rate = learning_rate
        self.eta0 = eta0

        self.power_t = power_t
        self.average = average

        self.model = SGDClassifier(loss=self.loss, penalty=self.penalty, alpha=self.alpha,
                                   l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept,
                                   tol=self.tol, epsilon=self.epsilon,
                                   learning_rate=self.learning_rate, eta0=self.eta0,
                                   power_t=self.power_t, average=self.average,
                                   early_stopping=True, random_state=self.seed)

    def fit(self, X: np.array, y: np.array):
        self.model.fit(X, y)

    def predict(self, X: np.array) -> np.array:
  
        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]
        return self.model.predict(X)

    def predict_proba(self, X: np.array) -> np.array:
        """
        Some losses of the sgd implementation of sklearn does not allow for predict_proba.
        To allow the return of probabilities compute the confidence values of the decision function
        and rescale them to pseudo probabilities using softmax.
        """
  
        # Need to check for the length, as else using RSM + CV will lead to problems
        if self.feature_mask is not None and X.shape[1] > len(self.feature_mask):
            X = X[:, self.feature_mask]

        confidence = self.model.decision_function(X)
        return softmax(confidence)

    def get_params(self, deep=True) -> dict:
        return {'loss': self.loss,
                'penalty': self.penalty,
                'alpha': self.alpha,
                'l1_ratio': self.l1_ratio,
                'fit_intercept': self.fit_intercept,
                'tol': self.tol,
                'epsilon': self.epsilon,
                'learning_rate': self.learning_rate,
                'eta0': self.eta0,
                'power_t': self.power_t,
                'average': self.average,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'SGD',
                'algorithm': CLASSIFICATION,
                'is_deterministic': False,
                'input': MIXED,
                'problem': (MULTICLASS,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        loss = CategoricalHyperparameter('loss', ['hinge', 'log_loss', 'modified_huber', 'squared_hinge',
                                                  'perceptron'], default_value='hinge')
        penalty = CategoricalHyperparameter('penalty', ['l1', 'l2', 'elasticnet'], default_value='l2')

        alpha = UniformFloatHyperparameter('alpha', lower=1e-6, upper=1e-1, log=True, default_value=1e-4)
        l1_ratio = UniformFloatHyperparameter('l1_ratio', lower=1e-4, upper=1.0, log=True, default_value=0.15)
        tol = UniformFloatHyperparameter('tol', lower=1e-5, upper=1e-1, log=True, default_value=1e-3)
        epsilon = UniformFloatHyperparameter('epsilon', lower=1e-4, upper=1e-1, log=True, default_value=0.1)

        learning_rate = CategoricalHyperparameter('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'], default_value='optimal')
        eta0 = UniformFloatHyperparameter('eta0', lower=1e-5, upper=1e-1, log=True, default_value=1e-5)
        power_t = UniformFloatHyperparameter('power_t', lower=1e-5, upper=1.0, default_value=0.5)
        average = CategoricalHyperparameter('average', [False, True], default_value=False)

        cond_l1 = InCondition(l1_ratio, penalty, ['l1'])
        cond_eta0 = InCondition(eta0, learning_rate, ['constant', 'invscaling', 'adaptive'])

        c_space.add_hyperparameters(
            [loss, penalty, alpha, l1_ratio, tol, epsilon, learning_rate, eta0, power_t, average])
        c_space.add_conditions([cond_l1, cond_eta0])

        return c_space
