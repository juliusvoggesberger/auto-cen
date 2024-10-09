"""
Implements a wrapper for the pusion MLP implementation.
"""

from ConfigSpace import EqualsCondition, InCondition, CategoricalHyperparameter, \
    UniformFloatHyperparameter, UnParametrizedHyperparameter, ConfigurationSpace, Constant

from auto_cen.constants import COMBINATION, LABELS, MULTICLASS, MULTILABEL, CONTINUOUS_OUT, \
    TRAINABLE_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import NeuralNetworkCombiner


class NeuralNetwork(BaseMethod):
    """
    Wrapper for the pusion combiner Neural Network.
    As only random search and no Neural Architecture Search is used, the Architecture is assumed to
    be static. The number of hidden layers is 3, as defined in the pusion Framework.
    The size of the layers is dependent on the given training data.

    Instead of "max_iter" the hyperparameter "tol" is used, as a stopping criterion for the solver
    iterations.

    :param activation: Activation function used for the hidden layers.
                       Either 'logistic', 'tanh' or 'relu'.
    :param solver: Solver used for the optimization of the weights. Either 'lbfgs', 'sgd' or 'adam'.
    :param alpha: Regularization term parameter. Value range [1e-5, 1e-2].
    :param tol: Stopping criterion. Value in range [1e-5, 1e-1].
    :param batch_size: Size of batches for SGD and Adam. Static value of 512. (Should ideally be a power of 2)
    :param learning_rate: Learning rate for updating weight parameters in SGD.
                          Either 'constant', 'invscaling' or 'adaptive'.
                          Only used with solver='sgd'.
    :param learning_rate_init: Initial learning rate.
                               Value in range [1e-4, 1e-2]. Only used for solver='sgd' or 'adam'.
    :param power_t: Exponent for invscaling. Value is 0.5. Only used for solver='sgd'.
    :param momentum: Momentum used for sgd updates. Value range [0.7, 0.99]. Only used for solver='sgd'.
    :param nesterovs_momentum: If True use nesterovs momentum. Only used for solver='sgd'.
    :param early_stopping: If True use early stopping. 10% of the training data are used for
                       validation. Only used for solver='sgd' or 'adam'. Always True.
    :param beta_1: Exponential decay rate for first moment vector. Value is 0.9.
                   Only used for solver='adam'.
    :param beta_2: Exponential decay rate for second moment vector. Value is 0.999.
                   Only used for solver='adam'.
    :param epsilon: Stabilization value for adam. Value is 1e-8. Only used for solver='adam'.
    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    :param seed: Random seed.
    """

    def __init__(self, activation: str, solver: str, alpha: float, tol: float,
                 batch_size: int = 512,
                 learning_rate: str = 'constant', learning_rate_init: float = 0.001,
                 power_t: float = 0.5,
                 momentum: float = 0.9, nesterovs_momentum: bool = True, beta_1: float = 0.9,
                 beta_2: float = 0.999, epsilon: float = 1e-8, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.activation = activation
        self.alpha = alpha
        self.tol = tol
        self.batch_size = batch_size

        self.solver = solver

        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t

        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.model = NeuralNetworkCombiner(activation=self.activation, solver=self.solver,
                                           alpha=self.alpha, batch_size=self.batch_size,
                                           tol=self.tol, learning_rate=self.learning_rate,
                                           learning_rate_init=self.learning_rate_init,
                                           power_t=self.power_t, momentum=self.momentum,
                                           nesterovs_momentum=self.nesterovs_momentum,
                                           early_stopping=True, beta_1=self.beta_1,
                                           beta_2=self.beta_2, epsilon=self.epsilon, seed=self.seed)

    def fit(self, X: list, y: list):
        self.model.train(X, y)

    def predict(self, X: list) -> list:
        return self.model.combine(X)

    def predict_proba(self, X: list) -> list:
        return self.model.combine(X, True)

    def get_params(self, deep=True) -> dict:
        return {'activation': self.activation,
                'solver': self.solver,
                'alpha': self.alpha,
                'tol': self.tol,
                'learning_rate': self.learning_rate,
                'learning_rate_init': self.learning_rate_init,
                'power_t': self.power_t,
                'momentum': self.momentum,
                'nesterovs_momentum': self.nesterovs_momentum,
                'beta_1': self.beta_1,
                'beta_2': self.beta_2,
                'epsilon': self.epsilon,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'NN',
                'algorithm': COMBINATION,
                'is_deterministic': False,
                'combiner_type': TRAINABLE_COMBINER,
                'input': (LABELS, CONTINUOUS_OUT),
                'problem': (MULTICLASS, MULTILABEL),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        # SGD unique hyperparameters USE DEFAULTS FOR SIMPLER PARAMETER SPACE!
        learning_rate = CategoricalHyperparameter('learning_rate',
                                                  ['constant', 'invscaling', 'adaptive'],
                                                  default_value='constant')
        # power_t = UniformFloatHyperparameter('power_t', 1e-5, 1)
        power_t = UnParametrizedHyperparameter('power_t', 0.5)
        momentum = UniformFloatHyperparameter('momentum', 0.7, 0.99, default_value=0.9)
        # momentum = UnParametrizedHyperparameter('momentum', 0.9)
        nesterovs_momentum = CategoricalHyperparameter('nesterovs_momentum', [True, False],
                                                       default_value=True)

        # Adam unique hyperparameters USE DEFAULTS FOR SIMPLER PARAMETER SPACE!
        beta_1 = UnParametrizedHyperparameter('beta_1', 0.9)
        beta_2 = UnParametrizedHyperparameter('beta_2', 0.999)
        # beta1 = UniformFloatHyperparameter('beta_1', 0.8, 0.999)
        # beta2 = UniformFloatHyperparameter('beta_2', 0.9, 0.9999)
        epsilon = UnParametrizedHyperparameter('epsilon', 1e-8)

        # Adam and SGD unique hyperparameters
        learning_rate_init = UniformFloatHyperparameter('learning_rate_init', lower=1e-4,
                                                        upper=1e-2, log=True, default_value=0.001)

        # Other hyperparameters
        # n_hidden_layers = UniformIntegerHyperparameter('n_hidden_layers', lower=1, upper=3)
        # n_neurons_layer = UniformIntegerHyperparameter('n_neurons_layer', lower=10, upper=200)
        solver = CategoricalHyperparameter('solver', ['lbfgs', 'sgd', 'adam'], default_value='adam')
        activation = CategoricalHyperparameter('activation', ['logistic', 'tanh', 'relu'],
                                               default_value='relu')
        alpha = UniformFloatHyperparameter('alpha', lower=1e-5, upper=1e-2, log=True,
                                           default_value=0.0001)
        tol = UniformFloatHyperparameter('tol', lower=1e-5, upper=1e-1, log=True,
                                         default_value=1e-4)

        # SGD conditions
        cond_rate = EqualsCondition(learning_rate, solver, 'sgd')
        cond_power_solver = EqualsCondition(power_t, solver, 'sgd')
        cond_power_learning = EqualsCondition(power_t, learning_rate, 'invscaling')
        cond_momentum = EqualsCondition(momentum, solver, 'sgd')
        cond_nesterovs = EqualsCondition(nesterovs_momentum, solver, 'sgd')

        # ADAM conditions
        cond_beta1 = EqualsCondition(beta_1, solver, 'adam')
        cond_beta2 = EqualsCondition(beta_2, solver, 'adam')
        cond_eps = EqualsCondition(epsilon, solver, 'adam')

        # SGD and ADAM conditions
        cond_init = InCondition(learning_rate_init, solver, ['sgd', 'adam'])

        c_space.add_hyperparameters(
            [solver, activation, alpha, tol, beta_1, beta_2,
             epsilon, learning_rate, power_t, momentum, nesterovs_momentum, learning_rate_init])
        c_space.add_conditions(
            [cond_nesterovs, cond_momentum, cond_power_solver, cond_power_learning, cond_rate,
             cond_beta1,
             cond_beta2, cond_eps, cond_init])

        return c_space
