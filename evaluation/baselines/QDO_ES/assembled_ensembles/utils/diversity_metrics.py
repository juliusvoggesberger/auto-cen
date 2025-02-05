"""
A module containing functions to compute different diversity metrics.
"""

import math
import numpy as np

from typing import Union
from assembled_ensembles.utils.constants import DOUBLEFAULT, DISAGREEMENT, YULES_Q, CORRELATION_COEFFICIENT, \
    MC_DOUBLEFAULT, MC_DISAGREEMENT, KAPPA, YULES_Q_NORM, DOUBLEFAULT_NORM, MC_DOUBLEFAULT_NORM, \
    CORRELATION_COEFFICIENT_NORM


def _fix_ratios(uniques, counts):
    """
    Checks if all possible combination scenarios are listed in the uniques list.
    If not, the missing scenarios are added as having 0 counts.

    :param uniques: The given unique scenarios
    :param counts: The given counts to each scenario
    :return The fixed counts for each scenario
    """
    new_uniques = [[False, False], [False, True], [True, False], [True, True]]
    new_counts = []
    count = 0
    for i in range(len(new_uniques)):
        if not np.any([new_uniques[i] == list(x) for x in uniques]):
            new_counts.append(0)
        else:
            new_counts.append(counts[count])
            count += 1
    return new_uniques, new_counts


def compare_two_classifiers_diff(diff_y1, diff_y2) -> (float, float, float, float):
    """
    Compare the true/false predictions of two classifiers c1 und c2.
    a = fraction of predictions where c1 and c2 are true
    b = fraction of predictions where c1 is true and c2 is false
    c = fraction of predictions where c1 is false and c2 is true
    d = fraction of predictions where c1 and c2 are false

    :param diff_y1: Difference between prediction of first classifier and ground truth.
    :param diff_y2: Difference between prediction of second classifier and ground truth.
    :return: a, b, c, d as a tuple of floats in that order
    """

    diff_matrix = np.column_stack((diff_y1, diff_y2))
    unis, counts = np.unique(diff_matrix, axis=0, return_counts=True)
    if len(unis) < 4:
        unis, counts = _fix_ratios(unis, counts)
    d, c, b, a = counts / np.sum(counts)
    return a, b, c, d

def compute_diversity_metrics(diff_1, diff_2, y1, y2, y_true, metrics: Union[str, list]) -> dict:
    """
    Computes a or multiple performance metric(s) for given data.
    The metrics are normalized to the range of [0,1], with 1 = high diversity and 0 = low diversity.

    WARNING: The kappa method depends on the order in which the classifiers are passed! USE WITH CARE!

    :param diff_1: Diff. of first classifier predictions against ground truth
    :param diff_2: Diff. of second classifier predictions against ground truth
    :param y1: Predictions of 1st classifier
    :param y2: Predictions of 2nd classifier
    :param y_true: Ground truth labels
    :param metrics: Either a string (one metric) or a list (multiple metrics).
    :return: A dict where the key is the metric and the value the metric score
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    scores = {}

    if np.array(y1).ndim > 1:  # Convert to labels, if prob_predictions
        y1 = np.argmax(y1, axis=1)
        y2 = np.argmax(y2, axis=1)

    if any(m in [DOUBLEFAULT_NORM, DOUBLEFAULT, DISAGREEMENT, YULES_Q, YULES_Q_NORM,
                 CORRELATION_COEFFICIENT, CORRELATION_COEFFICIENT_NORM, KAPPA] for m in
           metrics):
        # Need to compute these for oracle output metrics
        a, b, c, d = compare_two_classifiers_diff(diff_1, diff_2)
    if any(m in [MC_DOUBLEFAULT, MC_DISAGREEMENT] for m in metrics):
        diff = get_oracle_predictions(y1, y2)
    for m in metrics:
        if m == DOUBLEFAULT:
            scores[m] = double_fault(d)
        elif m == DOUBLEFAULT_NORM:
            scores[m] = double_fault_norm(d)
        elif m == DISAGREEMENT:
            scores[m] = disagreement(b, c)
        elif m == YULES_Q:
            scores[m] = yules_q(a, b, c, d)
        elif m == YULES_Q_NORM:
            scores[m] = yules_q_norm(a, b, c, d)
        elif m == CORRELATION_COEFFICIENT:
            scores[m] = correlation(a, b, c, d)
        elif m == CORRELATION_COEFFICIENT_NORM:
            scores[m] = correlation_norm(a, b, c, d)
        elif m == KAPPA:
            scores[m] = kappa_error_pairwise(a, b, c, d)
        elif m == MC_DISAGREEMENT:
            scores[m] = multiclass_disagreement(diff)
        elif m == MC_DOUBLEFAULT:
            scores[m] = multiclass_doublefault(diff, y1, y_true)
        elif m == MC_DOUBLEFAULT_NORM:
            scores[m] = multiclass_doublefault_norm(diff, y1, y_true)
    return scores


def correlation_norm(a: float, b: float, c: float, d: float) -> float:
    """
    Compute the correlation coefficient:

    .. math::
            \\rho_{i,j} = \\frac{ad-bc}{\\sqrt{(a+b) \\cdot (c+d) \\cdot (a+c) \\cdot (b+d)}}\\ .

    1 means low diversity, -1 means high diversity.
    This is because negative correlation of classifiers is wanted:
        See "L. Kuncheva, C. Whitaker, Measures of Diversity in Classifier Ensembles
        and Their Relationship with the Ensemble Accuracy, 2003.
    To standardize the results, the normal value range [-1,1] is normalized to [0,1], by adding 1 and dividing by 2.
    Now 0 is high diversity, 1 is low diversity.
    To standardize it compute 1-value, so high value = high diversity.

    As such the result is:

    .. math::
            result = 1 - \\frac{\\rho_{i,j} +1}{2}.

    :param a: Fraction of instances that are correctly classified by both classifiers.
    :param b: Fraction of instances that are correctly classified by the first classifier and wrongly by the second.
    :param c: Fraction of instances that are wrongly classified by the first classifier and correctly by the second.
    :param d: Fraction of instances that are wrongly classified by both classifiers.

    :return: Value of the correlation coefficient.
    """
    if (a + b == 0) or (c + d == 0) or (a + c == 0) or (b + d == 0):
        return 0
    corr = (a * d - b * c) / math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if corr > 1:
        # This should not happen, but could because of rounding errors.
        corr = 1.0
    return 1 - ((corr + 1) / 2)


def correlation(a: float, b: float, c: float, d: float) -> float:
    """
    Compute the correlation coefficient:

    .. math::
            \\rho_{i,j} = \\frac{ad-bc}{\\sqrt{(a+b) \\cdot (c+d) \\cdot (a+c) \\cdot (b+d)}}\\ .

    1 means low diversity, -1 means high diversity.
    This is because negative correlation of classifiers is wanted:
        See "L. Kuncheva, C. Whitaker, Measures of Diversity in Classifier Ensembles
        and Their Relationship with the Ensemble Accuracy, 2003.

    :param a: Fraction of instances that are correctly classified by both classifiers.
    :param b: Fraction of instances that are correctly classified by the first classifier and wrongly by the second.
    :param c: Fraction of instances that are wrongly classified by the first classifier and correctly by the second.
    :param d: Fraction of instances that are wrongly classified by both classifiers.

    :return: Value of the correlation coefficient.
    """
    if (a + b == 0) or (c + d == 0) or (a + c == 0) or (b + d == 0):
        return 0
    corr = (a * d - b * c) / math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if corr > 1:
        # This should not happen, but could because of rounding errors.
        corr = 1.0
    return corr


def yules_q_norm(a: float, b: float, c: float, d: float) -> float:
    """
    Compute yules_q:

    .. math::
            Q_{i,j} = \\frac{ad-bc}{ad+bc}\\ .

    1 means low diversity, -1 means high diversity.
    This is because negative correlation of classifiers is wanted:
        See "L. Kuncheva, C. Whitaker, Measures of Diversity in Classifier Ensembles
        and Their Relationship with the Ensemble Accuracy, 2003.
    To standardize the results, the normal value range [-1,1] is normalized to [0,1], by adding 1 and dividing by 2.
    Now 0 is high diversity, 1 is low diversity.
    To standardize it compute 1-value, so high value = high diversity.

    As such the result is:

    .. math::
            result = 1 - \\frac{\\rho_{i,j} +1}{2}.

    :param a: Fraction of instances that are correctly classified by both classifiers.
    :param b: Fraction of instances that are correctly classified by the first classifier and wrongly by the second.
    :param c: Fraction of instances that are wrongly classified by the first classifier and correctly by the second.
    :param d: Fraction of instances that are wrongly classified by both classifiers.

    :return: Value of yules Q.
    """
    if (a + b) == 0 or (a + c) == 0 or (d + b) == 0 or (d + c) == 0:
        return 0
    q = (a * d - b * c) / (a * d + b * c)
    return 1 - ((q + 1) / 2)


def yules_q(a: float, b: float, c: float, d: float) -> float:
    """
    Compute yules_q:

    .. math::
            Q_{i,j} = \\frac{ad-bc}{ad+bc}\\ .

    1 means low diversity, -1 means high diversity.
    This is because negative correlation of classifiers is wanted:
        See "L. Kuncheva, C. Whitaker, Measures of Diversity in Classifier Ensembles
        and Their Relationship with the Ensemble Accuracy, 2003.

    :param a: Fraction of instances that are correctly classified by both classifiers.
    :param b: Fraction of instances that are correctly classified by the first classifier and wrongly by the second.
    :param c: Fraction of instances that are wrongly classified by the first classifier and correctly by the second.
    :param d: Fraction of instances that are wrongly classified by both classifiers.

    :return: Value of yules Q.
    """
    if (a + b) == 0 or (a + c) == 0 or (d + b) == 0 or (d + c) == 0:
        return 0
    q = (a * d - b * c) / (a * d + b * c)
    return q


def disagreement(b: float, c: float) -> float:
    """
    Computes the disagreement metric.
    disagreement is computed by taking the fraction of samples, where both classifiers are disagreeing.
    I.e. when one classifier is wrong the other is right.
    The higher the value, the higher the diversity.

    :param b: Fraction of instances that are correctly classified by the first classifier and wrongly by the second.
    :param c: Fraction of instances that are wrongly classified by the first classifier and correctly by the second.

    :return: The disagreement value.
    """
    return b + c


def double_fault_norm(d: float) -> float:
    """
    Computes the double-fault metric.
    Double-fault is computed by taking the fraction of samples, where both classifiers are wrong.
    The higher the value, the lower the diversity.
    To standardize it, compute 1-double-fault as the result. Then high value = high diversity.

    :param d: Fraction of instances that are wrongly classified by both classifiers.

    :return: The double-fault value.
    """
    # High value = High ratio of similar errors = Low diversity -> High value = Low diversity
    return 1 - d


def double_fault(d: float) -> float:
    """
    Computes the double-fault metric.
    Double-fault is computed by taking the fraction of samples, where both classifiers are wrong.
    The higher the value, the lower the diversity.
    To standardize it, compute 1-double-fault as the result. Then high value = high diversity.

    :param d: Fraction of instances that are wrongly classified by both classifiers.

    :return: The double-fault value.
    """
    # High value = High ratio of similar errors = Low diversity -> High value = Low diversity
    return d


def kappa_error_pairwise(a: float, b: float, c: float, d: float) -> float:
    """
    Calculates the pairwise Interrater agreement (Kappa) metric for 2 Classifiers.
    Low values mean high diversity and vice versa.
    To standardize it compute 1-value, so high value = high diversity.
    WARNING: This method depends on the order in which the classifiers are passed! USE WITH CARE!

    .. math::
            \\kappa_{i,j} = \\frac{2\\cdot(ac-bd)}{(a+b)(c+d)+(a+c)(b+d)}.

    :param a: Fraction of instances that are correctly classified by both classifiers.
    :param b: Fraction of instances that are correctly classified by the first classifier and wrongly by the second.
    :param c: Fraction of instances that are wrongly classified by the first classifier and correctly by the second.
    :param d: Fraction of instances that are wrongly classified by both classifiers.

    :return: The kappa value.
    """
    kappa_numerator = 2 * ((a * c) - (b * d))
    kappa_denominator = (a + b) * (c + d) + (a + c) * (b + d)

    if kappa_denominator == 0:
        return 0
    kappa = (kappa_numerator / kappa_denominator)

    # The kappa CAN become negativ in special cases.
    # If that is the case cap it at 0, so we can use it as a metric.
    if kappa < 0:
        kappa = 0

    return kappa


def multiclass_doublefault_norm(diff, y1, y_true):
    """
    Adapted from:
    "Peter Bellmann, Patrick Thiam and Friedhelm Schwenker: Multi-classifier-Systems: Architectures, Algorithms and Applications, 2018"

    .. math::
            df = \\frac{1}{N} \\sum^N_{i=1} I_{c1(x_i) == c2(x_i)} \\cdot I_{c1(x_i) != y(x_i)}

    With N: Number of instance, :mat:`x_i`: The i-th instance, c1,c2: Both classifiers,
    :mat:`y(x_i)`: the ground truth of :mat:`x_i` and I: The characteristic function

    :param diff: Differences between the predictions of both classifiers
    :param y1: The prediction of one of the classifiers
    :param y_true: The ground truth
    :return: The multiclass double-fault metric
    """

    # Find the indices where both classifiers agree
    # Using the subset of instances where both classifiers agree, find the indices where they both are wrong
    # Sum these up
    df = (np.sum(y1[diff] != y_true[diff]) / len(y1))

    return 1 - df


def multiclass_doublefault(diff, y1, y_true):
    """
    Adapted from:
    "Peter Bellmann, Patrick Thiam and Friedhelm Schwenker: Multi-classifier-Systems: Architectures, Algorithms and Applications, 2018"

    .. math::
            df = \\frac{1}{N} \\sum^N_{i=1} I_{c1(x_i) == c2(x_i)} \\cdot I_{c1(x_i) != y(x_i)}

    With N: Number of instance, :mat:`x_i`: The i-th instance, c1,c2: Both classifiers,
    :mat:`y(x_i)`: the ground truth of :mat:`x_i` and I: The characteristic function

    :param diff: Differences between the predictions of both classifiers
    :param y1: The prediction of one of the classifiers
    :param y_true: The ground truth
    :return: The multiclass double-fault metric
    """

    # Find the indices where both classifiers agree
    # Using the subset of instances where both classifiers agree, find the indices where they both are wrong
    # Sum these up
    df = (np.sum(y1[diff] != y_true[diff]) / len(y1))

    return df


def multiclass_disagreement(diff):
    """
    Adapted from:
    "Peter Bellmann, Patrick Thiam and Friedhelm Schwenker: Multi-classifier-Systems: Architectures, Algorithms and Applications, 2018"

    .. math::
            dis = \\frac{1}{N} \\sum^N_{i=1} I_{c1(x_i) != c2(x_i)})}

    With N: Number of instance, :mat:`x_i`: The i-th instance, c1,c2: Both classifiers
     and I: The characteristic function

    :param diff: Differences between the predictions of both classifiers
    :return: The multiclass disagreement metric
    """

    return np.sum(np.invert(diff)) / len(diff)


def kohawi_wolpert_variance(clf_y: np.array) -> float:
    """
    Non-Pairwise diversity measure
    Range between 0.0 and 0.25, where high is diverse and low is non-diverse

    :param clf_y: Oracle outputs (correct/incorrect) for all classifiers
    :return: The computed kohavi-wolpert metric.
    """
    kw = 0
    L = clf_y.shape[0]  # Number of classifiers
    N = clf_y.shape[1]  # Number of samples
    for zj in clf_y.T:
        Yzj = zj.sum()  # Number of correct votes for class zj among all classifiers
        kw += Yzj * (L - Yzj)
    kw = kw / ((L ** 2) * N)
    return kw


def kappa_error(clf_y: np.array) -> float:
    """
    Calculates the non-pairwise Interrater agreement (Kappa) metric for more than 2 Classifiers.

    :param clf_y: Oracle outputs (correct/incorrect) for all classifiers
    :return: The kappa value.
    """
    L = clf_y.shape[0]  # Number of classifiers
    N = clf_y.shape[1]  # Number of samples
    p = (1.0 / (N * L)) * np.sum(clf_y)  # Average individual classification accuracy

    kappa_numerator = 0
    for zj in clf_y.T:
        Yzj = zj.sum()  # Number of correct votes for class zj among all classifiers
        kappa_numerator += Yzj * (L - Yzj)
    kappa_numerator = (1 / L) * kappa_numerator
    kappa_denominator = N * (L - 1) * p * (1 - p)

    if kappa_denominator == 0:
        return 1  # Meaning no diversity

    return 1 - (kappa_numerator / kappa_denominator)


def get_oracle_predictions(y: list, y_true: np.array) -> np.array:
    """
    Compute the difference between a prediction and a ground truth, i.e. the oracle output of a classifier (True=1, False=0)
    The output will be a boolean array, which is True if the prediction was True and else False.

    :param y: Prediction
    :param y_true: Ground truth
    :return: Boolean array
    """

    # Have to convert, as y is the prediction and predictions have to be saved as list in RunInfo
    y_pred = np.array(y)
    if np.array(y).ndim > 1 and y_true.ndim == 1:  # Convert to labels, if prob_predictions
        y_pred = np.argmax(y_pred, axis=1)

    y_true = np.array(y_true)
    if y_true.ndim > 1:
        y_diff = (y_pred == y_true).all(axis=1)
    else:
        y_diff = (y_pred == y_true)
    return y_diff
