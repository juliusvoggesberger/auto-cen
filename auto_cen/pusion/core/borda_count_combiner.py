from auto_cen.pusion.core.combiner import UtilityBasedCombiner
from auto_cen.pusion.utils.transformer import *
from auto_cen.pusion.utils.constants import *


class BordaCountCombiner(UtilityBasedCombiner):
    """
    The :class:`BordaCountCombiner` (BC) is a decision fusion method that establishes a ranking between label
    assignments for a sample. This ranking is implicitly given by continuous support outputs and is mapped to different
    amounts of votes (:math:`0` of :math:`L` votes for the lowest support, and :math:`L-1` votes for the highest one).
    A class with the highest sum of these votes (borda counts) across all classifiers is considered as a winner for the
    final decision.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT)
    ]

    SHORT_NAME = 'BC'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def combine(self, decision_tensor: np.array) -> np.array:
        """
        Combine decision outputs by the Borda Count (BC) method. Firstly, the continuous classification is mapped to a
        ranking with respect to available classes for each sample. Those rankings are then summed up across all
        classifiers to establish total votes (borda counts) for each class in a sample. The class with the highest
        number of borda counts is considered as decision fusion. Only continuous classification outputs are supported.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp class assignments which represents fused decisions.
                Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in
                ``decision_tensor`` input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        fused_decisions = np.zeros_like(decision_tensor[0], dtype=int)

        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            sort_indices = np.argsort(dp, axis=1)
            # Swap the 2nd dim indices and values, to give the votes to each class
            # I.e. the index is the vote (and is swapped to be a value,
            # and the value is the class (and is swapped to be an index)
            bc_dp = self.swap_index_and_values(sort_indices)
            fused_decisions[i, np.argmax(np.sum(bc_dp, axis=0))] = 1
        return fused_decisions

    @staticmethod
    def swap_index_and_values(m: np.array) -> np.array:
        """
        Swap the values of an index matrix in its rows
        -> The indices of the rows will be swapped against the values of said indices.
        I.e. a matrix of values [[0,1,2],[1,2,0]] will get swapped to [[0,1,2],[2,0,1]]

        :param m: The indices of a matrix for sorting it.
        :return:
        """
        s = np.zeros_like(m)
        for i in range(len(m)):
            for j in range(len(m[i])):
                s[i, m[i, j]] = j
        return s
