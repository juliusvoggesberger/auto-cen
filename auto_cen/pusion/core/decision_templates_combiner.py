from auto_cen.pusion.core.combiner import TrainableCombiner
from auto_cen.pusion.utils.transformer import *
from auto_cen.pusion.utils.constants import *

import math


# Modified!

class DecisionTemplatesCombiner(TrainableCombiner):
    """
    The :class:`DecisionTemplatesCombiner` (DT) is adopted from the decision fusion method originally proposed by
    Kuncheva :footcite:`kuncheva2014combining`. A decision template is the average matrix of all decision profiles,
    which correspond to samples of one specific class. A decision profile contains classification outputs from all
    classifiers for a sample in a row-wise fashion. The decision fusion is performed based on distance calculations
    between decision templates and the decision profile generated from the ensemble outputs.

    :param similarity: The similarity metric used to compute the distance.
                        Euclidean, Manhattan, Cosine and Symmetric Difference are supported.

    .. footbibliography::
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'DTEMP'

    def __init__(self, similarity: str = 'euclidean'):
        super().__init__()
        # all possible classification occurrences (class assignments) in training data
        self.distinct_class_assignments = None
        # decision templates according to kuncheva per distinct class assignment (aligned with
        # distinct_class_assignments)
        self.decision_templates = None
        self.similarity_metric = similarity

    def train(self, decision_tensor: np.array, true_assignments: np.array):
        """
        Train the Decision Templates Combiner model by precalculating decision templates from given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported. This procedure involves
        calculating means of decision profiles (decision templates) for each true class.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        if np.shape(decision_tensor)[1] != np.shape(true_assignments)[0]:
            raise TypeError(
                "True assignment vector dimension does not match the number of samples.")

        # represent outputs of multiple classifiers as a DP for each sample
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        self.distinct_class_assignments = np.unique(true_assignments, axis=0)

        self.decision_templates = np.zeros((len(self.distinct_class_assignments),
                                            np.shape(decision_profiles[0])[0],
                                            np.shape(decision_profiles[0])[1]))

        for i in range(len(self.distinct_class_assignments)):
            # calculate the mean decision profile (decision template) for each class assignment.
            label = self.distinct_class_assignments[i]
            label_indices = np.where(np.all(label == true_assignments, axis=1))[0]
            self.decision_templates[i] = np.average(decision_profiles[label_indices], axis=0)

    def combine(self, decision_tensor: np.array) -> np.array:
        """
        Combine decision outputs by using the Decision Templates method.
        Both continuous and crisp classification outputs are supported. Combining requires a trained
        :class:`DecisionTemplatesCombiner`.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by the minimum distance between decision profiles of ``decision_tensor`` and
                precalculated decision templates. Axis 0 represents samples and axis 1 the class assignments which
                are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        fused_decisions = np.zeros_like(decision_tensor[0])

        # Compute the euclidean distance between each DP (for a class assignment) and trained DT.
        # The class assignment associated with the DT with minimal distance to the DP
        # is considered as the fused decision.
        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            dist = np.empty(len(self.decision_templates))
            for j in range(len(self.decision_templates)):
                dt = self.decision_templates[j]
                dist[j] = self._compute_similarity(dp, dt)

            min_dist_label = self.distinct_class_assignments[dist.argmin()]
            fused_decisions[i] = min_dist_label
        return fused_decisions

    def _compute_similarity(self, dp: np.array, dt: np.array) -> float:
        dist = -1.
        if self.similarity_metric == 'euclidean':
            dist = np.average((dp - dt) ** 2)
        elif self.similarity_metric == 'manhattan':
            dist = np.average(np.abs(dp - dt))
        elif self.similarity_metric == 'cosine':
            dist = np.sum(dp * dt) / (np.linalg.norm(dp) * np.linalg.norm(dt))
        elif self.similarity_metric == 'symmetric_diff':
            dist = np.average(
                np.maximum(np.minimum(dt, 1 - dp), np.minimum((1 - dt), dp)))

        return dist

    def get_decision_templates(self) -> np.array:
        return self.decision_templates

    def get_distinct_class_assignments(self) -> np.array:
        return self.distinct_class_assignments
