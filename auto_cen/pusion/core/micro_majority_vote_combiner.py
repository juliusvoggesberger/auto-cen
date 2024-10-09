import numpy as np

from auto_cen.pusion.core.combiner import UtilityBasedCombiner
from auto_cen.pusion.utils.constants import *


# Modified!

class MicroMajorityVoteCombiner(UtilityBasedCombiner):
    """
    The :class:`MicroMajorityVoteCombiner` (MIMV) is based on a variation of the general majority vote method.
    The fusion consists of a decision vector which results from the majority of assignments for each individual class.

    :param threshold: A parameter between (0,1] for controlling the threshold for when a class is chosen.
                      I.e. if a class needs to have more then the thresholds votes.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'MIMV'

    def __init__(self, threshold: float = 0.5):
        """
        :param threshold: Threshold for the majority vote.
        """
        UtilityBasedCombiner.__init__(self)
        self.threshold = threshold

    def combine(self, decision_tensor: np.array) -> np.array:
        """
        Combine decision outputs by MIMV across all classifiers per class (micro).
        Only crisp classification outputs are supported.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different classifiers per sample.
                Crisp meaning that class labels are used.

        :return: A matrix (`numpy.array`) of crisp class assignments obtained by MIMV. Axis 0 represents samples and
                axis 1 the class labels which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        decision_tensor = decision_tensor - self.threshold
        decision_sum = np.sum(decision_tensor, axis=0)
        fused_decisions = (decision_sum >= 0) * np.ones_like(decision_sum)  # og (decision_sum > 0)?
        return fused_decisions
