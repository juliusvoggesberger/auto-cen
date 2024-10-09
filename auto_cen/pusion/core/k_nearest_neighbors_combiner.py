from sklearn.neighbors import KNeighborsClassifier

from auto_cen.pusion.core.combiner import TrainableCombiner
from auto_cen.pusion.utils.transformer import *
from auto_cen.pusion.utils.constants import *


class KNNCombiner(TrainableCombiner):
    """
    The :class:`KNNCombiner` (kNN) is a learning and classifier-based combiner that converts multiple decision
    outputs into new features, which in turn are used to train this combiner.
    The kNN combiner (k=5) uses uniform weights for all neighbors and the standard Euclidean metric for the distance.

    :param n_neighbors: The k parameter for the KNN algorithm.
    :param weights: Weighting used. Either 'uniform' or 'distance'
    :param p: Power parameter of minkowski distance. Either 1 (manhattan) or 2 (euclidean).
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'KNN_COMB'

    def __init__(self, n_neighbors: int, weights: str, p: int):
        TrainableCombiner.__init__(self)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)

    def train(self, decision_tensor: np.array, true_assignments: np.array):
        """
        Train the kNN combiner by fitting the `k` nearest neighbors (k=5) model with given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported.
        This procedure transforms decision outputs into a new feature space.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        self.classifier.fit(featured_decisions, true_assignments)

    def combine(self, decision_tensor: np.array, prob:bool = False) -> np.array:
        """
        Combine decision outputs by the `k` nearest neighbors (k=5) model.
        Both continuous and crisp classification outputs are supported. Combining requires a trained
        :class:`DecisionTreeCombiner`.
        This procedure transforms decision outputs into a new feature space.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.
        :param prob: If True, returns probability predictions. Default: False.

        :return: A matrix or tensor (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by kNN. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        if prob:
            return self.classifier.predict_proba(featured_decisions)
        else:
            return self.classifier.predict(featured_decisions)
