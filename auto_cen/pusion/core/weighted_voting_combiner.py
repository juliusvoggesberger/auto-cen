import numpy as np

from auto_cen.pusion.core.combiner import EvidenceBasedCombiner, TrainableCombiner
from auto_cen.pusion.utils.generator import *
from auto_cen.pusion.utils.constants import *


# Modified!

class WeightedVotingCombiner(EvidenceBasedCombiner, TrainableCombiner):
    """
    The :class:`WeightedVotingCombiner` (WV) is a weighted voting schema adopted from Kuncheva (eq. 4.43)
    :footcite:`kuncheva2014combining`. Classifiers with better performance (i.e. accuracy) are given more
    weight contributing to final decisions. Nevertheless, if classifiers of high performance disagree on a sample,
    low performance classifiers may contribute to the final decision.

    :param score_f: Scorer name. Can be ['accuracy', 'macro_recall', 'micro_recall', 'macro_precision',
                                                          'micro_precision', 'macro_f1', 'micro_f1']
    :param weight_f: Weighting function to be used. Can be ['simple', 'rescaled', 'best_worst', 'quadratic', 'logarithmic']
    :param n_classes: Number of classes of the problem. Needed for rescaled weighting.

    .. footbibliography::
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'WV'

    def __init__(self, score_f: str, weight_f: str, n_classes: int = 0):
        super().__init__()
        self.score_f = score_f
        self.weight_f = weight_f
        self.n_classes = n_classes
        self.score = None

    def set_evidence(self, evidence):
        """
        Set the evidence given by confusion matrices calculated according to Kuncheva :footcite:`kuncheva2014combining`
        for each ensemble classifier.

        .. footbibliography::

        :param evidence: `numpy.array` of shape `(n_classifiers, n_classes, n_classes)`.
                Confusion matrices for each of `n` classifiers.
        """
        self.score = _confusion_matrices_to_score_vector(self.score_f, evidence)

    def train(self, decision_tensor, true_assignments):
        """
        Train the Weighted Voting combiner model by precalculating confusion matrices from given decision outputs and
        true class assignments. Continuous decision outputs are converted into crisp multiclass assignments using
        the MAX rule.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        cms = generate_multiclass_confusion_matrices(decision_tensor, true_assignments)
        unweighted_score = _confusion_matrices_to_score_vector(self.score_f, cms)
        self.score = _weighting_function(self.weight_f, unweighted_score, self.n_classes)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by the weighted voting schema.
        Classifiers with better performance (i.e. accuracy) are given more authority over final decisions.
        Combining requires a trained :class:`WeightedVotingCombiner` or evidence set with ``set_evidence``.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp class assignments which represents fused
                decisions obtained by the maximum weighted class support. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        if self.score is None:
            raise TypeError("Accuracy is not set for this model as an evidence.")

        if np.shape(decision_tensor)[0] != np.shape(self.score)[0]:
            raise TypeError("Accuracy vector dimension does not match the number of classifiers in the input tensor.")
        # convert decision_tensor to decision profiles for better handling
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # average all decision profiles, i-th row contains decisions for i-th sample
        # Old implementation -> Only works for simple weighting
        # decision_matrix = np.array([np.average(dp, axis=0, weights=self.old_score) for dp in decision_profiles])
        dps = np.array([dp * self.score[:, None] for dp in decision_profiles])  # Multiply with weighted scores
        decision_matrix = np.array([dp.sum(axis=0) for dp in dps])  # Sum up over classifiers
        fused_decisions = np.zeros_like(decision_matrix)
        # find the maximum class support according to Kuncheva eq. (4.43)
        fused_decisions[np.arange(len(fused_decisions)), decision_matrix.argmax(axis=1)] = 1
        return fused_decisions


def _weighting_function(weighting: str, score: np.array, n_classes: int = 0) -> np.array:
    """
    Computes weights for the weighted voting combiner.
    Weights are taken from:
    F. Moreno-Seco, J. M. Inesta, P. J. P. De León, L. Micó.,
    „Comparison of classifier fusion methods for classification in pattern recognition tasks“,
    Springer, Berlin, Heidelberg, 2006.

    :param weighting: Weighting function to be used. Can be simple, rescaled, bw, bw_quadratic or logarithmic
    :param score: 1D Array of shape (n_classifiers) containing the evaluation score for each classifier
    :param n_classes: Number of classes, needed for "rescaled" weighting function
    :return: The weights
    """

    if weighting == "simple" or weighting == "rescaled":

        if weighting == "rescaled":
            # Set scores which are worse then random to 0.
            score[score < (1.0 / n_classes)] = 0

        score = np.divide(score, np.sum(score) * 1.0)
    elif weighting == "bw" or weighting == "bw_quadratic":
        best = np.max(score)
        worst = np.min(score)
        score = (score - worst) / (best - worst)
        if weighting == "bw_quadratic":
            np.power(score, 2)
    elif weighting == "logarithmic":
        # Natural logarithm
        score = np.log(np.divide(score, 1 - score))
    return score


def _confusion_matrices_to_score_vector(scorer, confusion_matrix_tensor) -> np.array:
    """
    Given a list of confusion matrices, convert them to a list of scores

    :param scorer: Scorer name
    :param confusion_matrix_tensor: Tensor that inhibits the confusion matrices
    :return: List of scores
    """
    scores = np.zeros(len(confusion_matrix_tensor))
    for i in range(len(confusion_matrix_tensor)):
        scores[i] = _confusion_matrix_to_score_vector(scorer, confusion_matrix_tensor[i])
    return scores


def _confusion_matrix_to_score_vector(scorer: str, cm: np.array) -> float:
    """
    Given a confusion matrix compute the given score.
    Supported scores are accuracy, recall (micro/macro), precision(micro/macro), f1(micro/macro),

    :param scorer: Scorer name
    :param cm: The confusion matrix
    :return: The score
    """
    cm = np.array(cm)
    tp = np.diagonal(cm)  # True positives

    if scorer == 'accuracy':
        return np.sum(np.diagonal(cm)) / np.sum(cm)

    # Micro
    if scorer in ['micro_recall', 'micro_precision', 'micro_f1']:
        micro_precision = np.sum(tp) / np.sum(np.sum(cm, axis=0))
        micro_recall = np.sum(tp) / np.sum(np.sum(cm, axis=1))
        if scorer == 'micro_recall':
            return micro_recall
        elif scorer == 'micro_precision':
            return micro_precision
        elif scorer == 'micro_f1':
            return 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    # Macro
    if scorer in ['macro_recall', 'macro_precision', 'macro_f1']:
        macro_p_sum = np.sum(cm, axis=0)
        macro_precision = np.sum(
            np.true_divide(tp, macro_p_sum, out=np.zeros_like(tp, dtype=float), where=macro_p_sum != 0)) / len(cm)
        macro_r_sum = np.sum(cm, axis=1)
        macro_recall = np.sum(
            np.divide(tp, macro_r_sum, out=np.zeros_like(tp, dtype=float), where=macro_r_sum != 0)) / len(cm)
        if scorer == 'macro_recall':
            return macro_recall
        elif scorer == 'macro_precision':
            return macro_precision
        elif scorer == 'macro_f1':
            return 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
