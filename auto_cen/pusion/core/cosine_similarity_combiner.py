import numpy as np

from scipy import spatial

from auto_cen.pusion.core.combiner import UtilityBasedCombiner
from auto_cen.pusion.utils.transformer import decision_tensor_to_decision_profiles
from auto_cen.pusion.utils.constants import *


class CosineSimilarityCombiner(UtilityBasedCombiner):
    """
    The :class:`CosineSimilarityCombiner` considers the classification assignments to :math:`\\ell` classes as vectors
    from an :math:`\\ell`-dimensional vector space. The normalized cosine-similarity measure between two vectors
    :math:`x` and :math:`y` is calculated as

    .. math::
            cos(x,y) = \\dfrac{x\\cdot y}{|x||y|}\\ .

    The cosine-similarity is calculated pairwise and accumulated for each classifier for one specific sample.
    The fusion is represented by a classifier which shows the most similar classification output to the output of all
    competing classifiers.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT)
    ]

    SHORT_NAME = 'COS'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def combine(self, decision_tensor: np.array) -> np.array:
        """
        Combine decision outputs with as an output that accommodates the highest cosine-similarity to the output of
        all competing classifiers. In other words, the best representative classification output among the others is
        selected according to the highest cumulative cosine-similarity. This method supports both, continuous and
        crisp classification outputs.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.


        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by the highest cumulative cosine-similarity. Axis 0 represents samples and axis 1 the
                class labels which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        fused_decisions = np.zeros_like(decision_tensor[0])
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            accumulated_cos_sim = np.zeros(len(dp))
            for j in range(len(dp)):
                for k in range(len(dp)):
                    if j != k and np.any(dp[j]) and np.any(dp[k]):
                        # Calculate the cosine distance (assumption: no zero elements)
                        accumulated_cos_sim[j] = accumulated_cos_sim[j] + (1 - spatial.distance.cosine(dp[j], dp[k]))
            fused_decisions[i] = dp[np.argmax(accumulated_cos_sim)]
        return fused_decisions
