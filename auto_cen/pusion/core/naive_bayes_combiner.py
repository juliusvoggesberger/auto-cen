from auto_cen.pusion.core.combiner import TrainableCombiner, EvidenceBasedCombiner
from auto_cen.pusion.utils.generator import *
from auto_cen.pusion.utils.constants import *


class NaiveBayesCombiner(EvidenceBasedCombiner, TrainableCombiner):
    """
    The :class:`NaiveBayesCombiner` (NB) is a fusion method based on the Bayes theorem which is applied according to
    Kuncheva :footcite:`kuncheva2014combining` and Titterington et al. :footcite:`titterington1981comparison`.
    NB uses the confusion matrix as an evidence to calculate the a-priori probability and the bayesian belief value,
    which in turn the decision fusion bases on. NB requires outputs from uncorrelated classifiers in the ensemble.

    .. footbibliography::
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT)
    ]

    SHORT_NAME = 'NB'

    def __init__(self):
        TrainableCombiner.__init__(self)
        self.confusion_matrices = None
        # Number of samples per class
        self.n_samples_per_class = None

    def set_evidence(self, evidence):
        """
        Set the evidence given by confusion matrices calculated according to Kuncheva :footcite:`kuncheva2014combining`
        for each ensemble classifier.

        .. footbibliography::

        :param evidence: `numpy.array` of shape `(n_classifiers, n_classes, n_classes)`.
                Confusion matrices for each of `n` classifiers.
        """
        self.confusion_matrices = evidence
        self.n_samples_per_class = np.sum(evidence[0], axis=1)

    def train(self, decision_tensor, true_assignments):
        """
        Train the Naive Bayes combiner model by precalculating confusion matrices from given decision outputs and
        true class assignments. Continuous decision outputs are converted into crisp multiclass assignments using
        the MAX rule.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        self.confusion_matrices = generate_multiclass_confusion_matrices(decision_tensor, true_assignments)
        # Number of samples of certain class (multiclass case)
        self.n_samples_per_class = np.sum(true_assignments, axis=0)
        # self.sum_of_samples_per_class = np.sum(confusion_matrices[0], axis=1)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by using the Naive Bayes method.
        Continuous decision outputs are converted to crisp multiclass predictions using the MAX rule.
        Combining requires a trained :class:`NaiveBayesCombiner` or evidence set with ``set_evidence``.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp class assignments which represents fused
                decisions obtained by the maximum class support. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        if self.confusion_matrices is None or self.n_samples_per_class is None:
            raise RuntimeError("Untrained model.")
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        fused_decisions = np.zeros_like(decision_tensor[0])
        for i in range(len(decision_profiles)):
            # transform to a multiclass decision profile
            dp = multiclass_predictions_to_decisions(decision_profiles[i])
            n_classes = dp.shape[1]
            mu = np.zeros(n_classes)
            for j in range(n_classes):
                n_j = self.n_samples_per_class[j]
                mu[j] = n_j / len(decision_profiles)
                for k in range(dp.shape[0]):
                    mu[j] = mu[j] * ((self.confusion_matrices[k, j, np.argmax(dp[k])] + 1/n_classes) / (n_j + 1))
            fused_decisions[i, np.argmax(mu)] = 1
        return fused_decisions
