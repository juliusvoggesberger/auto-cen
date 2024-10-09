from sklearn.neural_network import MLPClassifier

from auto_cen.pusion.core.combiner import TrainableCombiner
from auto_cen.pusion.utils.transformer import *
from auto_cen.pusion.utils.constants import *


class NeuralNetworkCombiner(TrainableCombiner):
    """
    The :class:`NeuralNetworkCombiner` (NN) is a learning and classifier-based combiner that converts multiple decision
    outputs into new features, which in turn are used to train this combiner. The NN includes three hidden layers and a
    dynamic number of neurons per layer, which is given by (`n_classifiers * n_classes`).

    :param activation: Activation function used for the hidden layers. Either 'identity', 'logistic', 'tanh' or 'relu'.
    :param solver: Solver used for the optimization of the weights. Either 'lbfgs', 'sgd' or 'adam'.
    :param alpha: Regularization term parameter. Value range [1e-7, 1e-1].
    :param tol: Stopping criterion. Value range [1e-5, 1e-1].
    :param learning_rate: Learning rate for updating weight parameters in SGD.
                          Either 'constant', 'invscaling' or 'adaptive'. Only used with solver='sgd'.
    :param learning_rate_init: Initial learning rate.
                               Value in range [1e-7, 1e-1]. Only used for solver='sgd' or 'adam'.
    :param power_t: Exponent for invscaling. Value range [1e-5, 1]. Only used for solver='sgd'.
    :param momentum: Momentum used for sgd updates. Value in range [0,1]. Only used for solver='sgd'.
    :param nesterovs_momentum: If True use nesterovs momentum. Only used for solver='sgd'.
    :param early_stopping: If True use early stopping. 10% of the training data are used for validation.
                           Only used for solver='sgd' or 'adam'.
    :param beta_1: Exponential decay rate for first moment vector. Value in range [0.0, 0.9999999].
                   Only used for solver='adam'.
    :param beta_2: Exponential decay rate for second moment vector. Value in range [0.0, 0.9999999].
                   Only used for solver='adam'.
    :param epsilon: Stabilization value for adam. Value in range [1e-9, 1e-1]. Only used for solver='adam'.
    :param seed: Random seed.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'NN'

    N_LAYERS = 3

    def __init__(self, activation: str, solver: str, alpha: float, batch_size: int, tol: float,
                 learning_rate: str, learning_rate_init: float, power_t: float, momentum: float,
                 nesterovs_momentum: bool, early_stopping: bool, beta_1: float, beta_2: float,
                 epsilon: float, seed=None):
        TrainableCombiner.__init__(self)

        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.tol = tol

        self.solver = solver

        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t

        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.seed = seed
        self.classifier = None

    def train(self, decision_tensor, true_assignments):
        """
        Train the NN combiner by fitting the Neural Network model with given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported.
        This procedure transforms decision outputs into a new feature space.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        hidden_layer_sizes = (true_assignments.shape[1] * len(decision_tensor),) * self.N_LAYERS
        self.classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=self.activation,
                                        solver=self.solver, alpha=self.alpha, batch_size=self.batch_size,
                                        tol=self.tol, learning_rate=self.learning_rate,
                                        learning_rate_init=self.learning_rate_init, power_t=self.power_t,
                                        momentum=self.momentum, nesterovs_momentum=self.nesterovs_momentum,
                                        early_stopping=self.early_stopping, beta_1=self.beta_1, beta_2=self.beta_2,
                                        epsilon=self.epsilon, random_state=self.seed)

        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        self.classifier.fit(featured_decisions, true_assignments)

    def combine(self, decision_tensor, prob=False):
        """
        Combine decision outputs by the trained Neural Network model. Both continuous and crisp classification outputs
        are supported. Combining requires a trained :class:`NeuralNetworkCombiner`. This procedure transforms decision
        outputs into a new feature space.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.
        :param prob: If True, returns probability predictions. Default: False.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by NN. Axis 0 represents samples and axis 1 the class assignments which are aligned
                with axis 2 in ``decision_tensor`` input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        if prob:
            return self.classifier.predict_proba(featured_decisions)
        else:
            return self.classifier.predict(featured_decisions)
