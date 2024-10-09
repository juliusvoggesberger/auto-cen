import numpy as np

from auto_cen.pusion.core.combiner import UtilityBasedCombiner
from auto_cen.pusion.utils.transformer import multilabel_predictions_to_decisions
from auto_cen.pusion.utils.constants import *


class SimpleAverageCombiner(UtilityBasedCombiner):
    """
    The :class:`SimpleAverageCombiner` (AVG) fuses decisions using the arithmetic mean rule.
    The mean is calculated between decision vectors obtained by multiple ensemble classifiers for a sample.
    The AVG combiner is unaware of the input problem (multiclass/multilabel) or the assignment type (crisp/continuous).
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'AVG'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def combine(self, decision_tensor, prob=False):
        """
        Combine decision outputs by averaging the class support of each classifier in the given ensemble.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.
        :param prob: If True, returns probability predictions. Default: False.

        :return: A matrix (`numpy.array`) of crisp assignments which represents fused
                decisions obtained by the AVG method. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        if prob:
            return np.mean(decision_tensor, axis=0)
        else:
            return multilabel_predictions_to_decisions(np.mean(decision_tensor, axis=0), .5)
