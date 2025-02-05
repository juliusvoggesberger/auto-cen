"""
A module containing functions to compute different performance metrics for evaluating classifiers.
"""
from typing import Union
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score, jaccard_score, roc_curve, auc, precision_recall_curve, \
    average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from utils.constants import ACCURACY, MC, BALANCED_ACCURACY, AP_MACRO, AP_MICRO, ROC_AUC_OVO, \
    ROC_AUC_OVR
from utils.transformer import transform_label_vector_to_class_assignment_matrix, \
    multiclass_assignments_to_labels


def accuracy(y_true: np.array, y_pred: np.array) -> float:
    """
    Computes the accuracy score.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,).
                   In multilabel case a matrix of shape (n_samples, n_labels).
                   Labels have to be binary in the multilabel case.
    :param y_pred: Classifier predictions.
    :return: The accuracy score.
    """
    return accuracy_score(y_true, y_pred)


def balanced_accuracy(y_true: np.array, y_pred: np.array) -> float:
    """
    Computes the balanced accuracy score.
    Does not support the multilabel case.

    Balanced accuracy is computed as the average of the sum of the recall metric over all classes.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,).
    :param y_pred: Classifier predictions.
    :return: The balanced accuracy score.
    """
    return balanced_accuracy_score(y_true, y_pred)

def precision(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Computes the precision score for each class separately.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,).
                   In multilabel case a matrix of shape (n_samples, n_labels).
                   Labels have to be binary in the multilabel case.
    :param y_pred: Classifier predictions.
    :return: An array of shape (n_classes,) containing the precision score ofr each class.
    """
    return precision_score(y_true, y_pred, average=None, zero_division=0)


def recall(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Computes the recall score for each class separately.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,).
                   In multilabel case a matrix of shape (n_samples, n_labels).
                   Labels have to be binary in the multilabel case.
    :param y_pred: Classifier predictions.
    :return: An array of shape (n_classes,) containing the recall score ofr each class.
    """
    return recall_score(y_true, y_pred, average=None, zero_division=0)


def pr_curve(y_true: np.array, y_pred_prob: np.array, n_classes: int) -> list:
    """
    Computes the values for the precision recall curve.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,).
                   In multilabel case a matrix of shape (n_samples, n_labels).
                   Labels have to be binary in the multilabel case.
    :param y_pred_prob: Classifier probabilities for each class.
    :param n_classes: Number of classes of the classification problem
    :return: list of precision and recall values
    """
    if len(y_pred_prob.shape) == 1:
        # If labels are given instead of prob., transform to a class matrix (n_samples, n_classes)
        y_pred_prob = transform_label_vector_to_class_assignment_matrix(y_pred_prob, n_classes=n_classes)

    if n_classes > 2 and len(y_true.shape) == 1:
        # If labels are given instead of prob., transform to a class matrix (n_samples, n_classes)
        y_true = transform_label_vector_to_class_assignment_matrix(y_true, n_classes=n_classes)
    elif n_classes == 2:
        y_true = np.dstack((y_true, y_true))[0]

    pre, rec = {}, {}
    for i in range(n_classes):
        # Compute the precision, recall values for each class
        pre[i], rec[i], _ = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
    return [pre, rec]


def average_precision(y_true: np.array, y_pred_prob: np.array, average="micro") -> np.array:
    """
        Computes the average precision (AP) for the precision-recall curve.
        Uses either micro or macro averaging

        :param y_true: Ground truth labels.
                       In binary/multiclass case a vector of shape (n_samples,).
                       In multilabel case a matrix of shape (n_samples, n_labels).
                       Labels have to be binary in the multilabel case.
        :param y_pred_prob: Classifier probabilities for each class.
        :param average: Use micro or macro averaging. Default: "micro"
        :return: The AP score.
        """

    if len(y_true.shape) == 1:
        # If labels are given instead of prob., transform to a class matrix (n_samples, n_classes)
        y_true = transform_label_vector_to_class_assignment_matrix(y_true, len(y_pred_prob[0]))

    pre = np.zeros(y_pred_prob.shape[1])
    for i in range(y_pred_prob.shape[1]):
        # Compute the average precision score for each class
        pre[i] = average_precision_score(y_true[:, i], y_pred_prob[:, i], average=average)
    return pre


def roc(y_true: np.array, y_pred_prob: np.array, n_classes: int) -> (dict, dict, dict):
    """
    Computes different receiver operating characteristic metrics class-wise.
    The false-positive-rate, the true-positive-rate, the auc-score.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,).
    :param y_pred_prob: Classifier probabilities for each class.
    :param n_classes: Number of classes.
    :return: A dict for each the false-positive-rate, the true-positive-rate, the auc-score,
            where the keys are the class indices.
    """
    if n_classes > 2:
        return _roc_multiclass(y_true, y_pred_prob, n_classes)
    return _roc_binary(y_true, y_pred_prob)


def roc_auc(y_true: np.array, y_pred_prob: np.array, n_classes: int, average: str = "weighted",
            multiclass: str = "ovr") -> float:
    """

    Computes the Area under Curve for the Receiver Operator Characteristic.
    If a label vector (n_samples,) is passed for a multiclass case it will be transformed to a
    matrix (n_samples, n_classes) by encoding it in a One-vs-Rest fashion.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,).
    :param y_pred_prob: Classifier probabilities for each class.
    :param n_classes: Number of classes.
    :param average: Type of averaging to be performed.
                    Can be ‘micro’, ‘macro’, ‘samples’, ‘weighted’ or 'None'.
                    If 'None', the scores for each class will be returned.
                    Default is 'weighted'.
    :param multiclass: How to compute the multiclass case. Either 'ovo' or 'ovr'. Default is 'ovr'.

    :return: The roc auc score. If it is -1 something went wrong.
    """

    if n_classes > 2 and len(y_pred_prob.shape) < 2:
        # If multiclass and a label vector is passed: Transform it to a label matrix
        y_pred_prob = label_binarize(y_pred_prob, classes=list(range(n_classes)))

    try:
        ras = roc_auc_score(y_true, y_pred_prob, average=average, multi_class=multiclass,
                            labels=list(range(n_classes)))
    except ValueError:
        print(
            "Classes seem highly imbalanced, as at least one class was missing from 'y_true'. "
            "Try to use 'OVR' instead of 'OVO'.")
        ras = -1
    return ras


def _roc_binary(y_true: np.array, y_pred_prob: np.array) -> (dict, dict, dict):
    """
    Computes roc for binary problems.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,).
    :param y_pred_prob: Classifier probabilities for each class.
    :return: A dict for each the false-positive-rate, the true-positive-rate, the auc-score,
                where the keys are the class indices.
    """
    fpr, tpr, roc_auc = {}, {}, {}
    if len(y_true.shape) > 1:
        # Multilabel
        y_true = multiclass_assignments_to_labels(y_true)
    if len(y_pred_prob.shape) > 1:
        # Assume second label (1) is the positive one
        y_pred_prob = y_pred_prob[:, 1]
    fpr[0], tpr[0], _ = roc_curve(y_true, y_pred_prob)
    roc_auc[0] = auc(fpr[0], tpr[0])
    return fpr, tpr, roc_auc


def _roc_multiclass(y_true: np.array, y_pred_prob: np.array, n_classes: int) -> (dict, dict, dict):
    """
    Computes roc for multiclass problems.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,).
    :param y_pred_prob: Classifier probabilities for each class.
    :param n_classes: Number of classes.
    :return: A dict for each the false-positive-rate, the true-positive-rate, the auc-score,
            where the keys are the class indices.
    """
    fpr, tpr, roc_auc = {}, {}, {}
    y_true = label_binarize(y_true, classes=list(range(n_classes)))
    if len(y_pred_prob.shape) == 1:
        # Convert to matrix -> Is the case, if a label vector, not a probability vector is passed
        y_pred_prob = transform_label_vector_to_class_assignment_matrix(y_pred_prob, n_classes)
    for i in range(n_classes):
        # Compute fpr, tpr, roc for each class
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


def mean_confidence(y_true: np.array, y_pred_prob: np.array) -> float:
    """
    Computes the mean confidence score, as introduced in
    A. Obralija. "Framework zur Entscheidungsfusion für die Kombination von Fehlerdiagnosemethoden".
    Only defined for binary/multiclass case.

    .. math::
            MC = 1 - \\frac{1}{n_{classes} \\cdot n_{samples}}
                \\sum_{i=1}^{n_{samples}} \\sum_{j=1}^{n_{classes}} |Y_{i,j} - T_{i,j}|\\.

    With Y=y_pred_prob and T=y_true.

    :param y_true: Ground truth labels.
                   In binary/multiclass case a vector of shape (n_samples,)
                   or a matrix of shape (n_samples, n_classes).
    :param y_pred_prob: Classifier probabilities for each class.
    :return: The mean confidence score.
    """
    if len(y_true.shape) < 2 and y_true.shape != y_pred_prob.shape:
        y_true = transform_label_vector_to_class_assignment_matrix(y_true, y_pred_prob.shape[1])

    return 1 - (np.sum(np.abs(y_true - y_pred_prob)) / (
            y_true.shape[0] * y_true.shape[1]))


def _compute_confidence_scores(y_true: np.array, y_pred_prob: np.array,
                               metrics: Union[str, list]) -> dict:
    """
    Computes one or multiple metric(s) for probability predictions.

    :param y_true: Ground truth labels.
    :param y_pred_prob: Probabilities for all labels.
    :param metrics: The metric(s) that will be computed.
    :return: A dict where the key is the metric and the value the metric score.
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    scores = {}

    for metric in metrics:
        try:
            if metric in [MC, AP_MICRO, AP_MACRO] and (
                    y_pred_prob is None or len(y_pred_prob.shape) < 2):
                scores[metric] = 0
            elif metric == MC:
                scores[metric] = mean_confidence(y_true, y_pred_prob)
            elif metric == AP_MICRO:
                scores[metric] = average_precision(y_true, y_pred_prob, average="micro")
            elif metric == AP_MACRO:
                scores[metric] = average_precision(y_true, y_pred_prob, average="macro")
        except ValueError:
            scores[metric] = 0
    return scores


def _compute_performance_metrics(y_true: np.array, y_pred: np.array,
                                 metrics: Union[str, list], n_classes: int, labels=None) -> dict:
    """
    Computes one or multiple performance metric(s) for given data.

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :param metrics: Either a string (one metric) or a list (multiple metrics)
    :param labels: Labels to be included for class-wise metrics
    :return: A dict where the key is the metric and the value the metric score.
    """

    if len(y_true.shape) > 1:
        # If prob predictions and/or this is a fusion prediction, the fusion ground truth
        # Has to be reduced to 1d
        y_true = np.argmax(y_true, axis=1)

    if len(y_pred.shape) > 1 and len(y_true.shape) == 1:
        # Transform the probability predictions/fusion predictions to label predictions
        y_pred = np.argmax(y_pred, axis=1)
    # Labels are needed for scoring functions.
    if isinstance(metrics, str):
        metrics = [metrics]
    scores = {}

    for metric in metrics:

        if metric == ACCURACY:
            scores[metric] = accuracy(y_true, y_pred)
        elif metric == BALANCED_ACCURACY:
            print(BALANCED_ACCURACY)
            scores[metric] = balanced_accuracy(y_true, y_pred)
        elif metric == ROC_AUC_OVO:
            scores[metric] = roc_auc(y_true, y_pred, n_classes, multiclass="ovo")
        elif metric == ROC_AUC_OVR:
            scores[metric] = roc_auc(y_true, y_pred, n_classes, multiclass="ovr")
        else:
            average = None
            if "macro" in metric:
                average = "macro"
            elif "micro" in metric:
                average = "micro"

            if "f1" in metric:
                scores[metric] = f1_score(y_true, y_pred, average=average, labels=labels)
            elif "precision" in metric:
                scores[metric] = precision_score(y_true, y_pred, zero_division=0, average=average,
                                                 labels=labels)
            elif "recall" in metric:
                print("recall")
                scores[metric] = recall_score(y_true, y_pred, average=average, labels=labels)
            elif "jaccard" in metric:
                scores[metric] = jaccard_score(y_true, y_pred, average=average, labels=labels)

    return scores


def create_performance_report(y_true: np.array, y_pred: np.array, y_prob: np.array,
                              metrics: Union[str, list], n_classes: int, labels=None) -> dict:
    """
    Computes both performance and confidence scores.

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :param y_prob: Probabilities for all labels.
    :param metrics: Either a string (one metric) or a list (multiple metrics)
    :param labels: Labels to be included for class-wise metrics
    :return: A dict where the key is the metric and the value the metric score.
    """
    scores = _compute_performance_metrics(y_true, y_pred, metrics, n_classes, labels)
    scores.update(_compute_confidence_scores(y_true, y_prob, metrics))
    return scores
