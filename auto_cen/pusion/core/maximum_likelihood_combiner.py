from auto_cen.pusion.core.combiner import TrainableCombiner
from auto_cen.pusion.utils.transformer import *
from auto_cen.pusion.utils.constants import *


class MaximumLikelihoodCombiner(TrainableCombiner):
    """
    The :class:`MaximumLikelihoodCombiner` (MLE) is a combiner that estimates the parameters :math:`\\mu` (sample means)
    and :math:`\\sigma` (sample variances) of the Gaussian probability density function for each class :math:`\\omega`.
    Multiple decision outputs for a sample are converted into a new feature space.

    The fusion is performed by evaluating the class conditional density

    .. math::
        p(x|\\omega) = \\frac{1}{\\sigma \\sqrt{2 \\pi}}
            exp\\left({-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}\\right).

    of a transformed sample :math:`x` for each available class :math:`\\omega`, respectively. The class with the highest
    likelihood is considered as winner and thus forms the decision fusion.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'MLE'

    def __init__(self):
        TrainableCombiner.__init__(self)
        self.unique_assignments = None
        self.mu = []
        self.sigma = []

    def train(self, decision_tensor: np.array, true_assignments: np.array):
        """
        Train the Maximum Likelihood combiner model by calculating the parameters of gaussian normal distribution
        (i.e. means and variances) from the given decision outputs and true class assignments.
        Both continuous and crisp classification outputs are supported. This procedure transforms decision outputs
        into a new feature space.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        # extract all occurring classifications
        self.unique_assignments, unique_inv_indices = np.unique(true_assignments, axis=0, return_inverse=True)

        # calculate the parameters for the multivariate normal distribution
        for i in range(len(self.unique_assignments)):
            xc = featured_decisions[np.where(unique_inv_indices == i)]  # X_train(class)
            mu = np.mean(xc, axis=0)
            sigma = np.std(xc, axis=0)
            sigma[sigma == 0] = 0.00001  # Add a small perturbation in order to enable class conditional density

            self.mu.append(mu)
            self.sigma.append(sigma)

    def combine(self, decision_tensor: np.array) ->np.array:
        """
        Combine decision outputs by the Maximum Likelihood method. This procedure involves evaluating the class
        conditional density as described above. Both continuous and crisp classification outputs are supported.
        Combining requires a trained :class:`MaximumLikelihoodCombiner`.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by MLE. Axis 0 represents samples and axis 1 the class assignments which are aligned
                with axis 2 in ``decision_tensor`` input tensor.
        """
        fused_decisions = np.zeros_like(decision_tensor[0])
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        import warnings
        warnings.filterwarnings('error')
        for i in range(len(featured_decisions)):
            x = featured_decisions[i]
            likelihoods = np.ones(len(self.unique_assignments))
            for j in range(len(self.unique_assignments)):
                for k in range(len(x)):
                    # calculate class conditional density for each dimension
                    try:
                        exp = (x[k] - self.mu[j][k]) / self.sigma[j][k]
                        exp_h = np.e ** (-.5 * (exp ** 2.0))
                        likelihoods[j] = likelihoods[j] * 1.0 / (np.sqrt(2.0 * np.pi) * self.sigma[j][k]) * exp_h
                    except RuntimeWarning: # TODO
                        # Not nice, but likelihoods throws runtime warnings
                        pass
            fused_decisions[i] = self.unique_assignments[np.argmax(likelihoods)]
        return fused_decisions

