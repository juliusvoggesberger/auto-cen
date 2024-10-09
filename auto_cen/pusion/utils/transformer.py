import numpy as np


def decision_tensor_to_decision_profiles(decision_tensor: np.array) -> np.array:
    """
    Transform the given decision tensor to decision profiles for each respective sample.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of either crisp or continuous decision outputs by different classifiers per sample.
    :return: `numpy.array` of shape `(n_samples, n_classifiers, n_classes)`.
            Decision profiles.
    """
    return decision_tensor.transpose((1, 0, 2))


def multilabel_predictions_to_decisions(predictions: np.array, threshold: float = 0.5) -> np.array:
    """
    Transform a continuously valued tensor of multilabel decisions to crisp decision outputs.

    :param predictions: `numpy.array` of any shape. Continuous predictions.
    :param threshold: `float`. A threshold value, based on which the crisp output is constructed.
    :return: `numpy.array` of the same shape as ``predictions``. Crisp decision outputs.
    """
    return (predictions >= threshold) * np.ones_like(predictions)


def multiclass_predictions_to_decisions(predictions: np.array) -> np.array:
    """
    Transform a continuously valued matrix of multiclass decisions to crisp decision outputs.

    :param predictions: `numpy.array` of shape `(n_samples, n_classes)`. Continuous predictions.
    :return: `numpy.array` of the same shape as ``predictions``. Crisp decision outputs.
    """
    if predictions.ndim == 1:
        return predictions
    decisions = np.zeros_like(predictions)
    decisions[np.arange(len(decisions)), predictions.argmax(axis=1)] = 1
    return decisions


def multiclass_assignments_to_labels(assignments: np.array) -> np.array:
    """
    Transform multiclass assignments to labels. A matrix of shape `(n_samples, n_classes)` is converted to a vector
    of shape `(n_samples,)`, with element-wise labels represented in integers from `0` to `n_classes - 1`.
    :param assignments: `numpy.array` of shape `(n_samples, n_classes)`. Multiclass assignments.
    :return: `numpy.array` of shape `(n_samples,)` with an integer label per element.
    """
    return np.argmax(assignments, axis=assignments.ndim - 1)


def decision_tensor_to_configs(decision_outputs: np.array) -> np.array:
    """
    Transform crisp decision outputs to decision configs.
    A decision config shows concatenated classification outputs of each classifier per sample.

    :param decision_outputs: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
    :return: `numpy.array` of shape `(n_samples, n_classes*)`, `n_classes*` is the concatenation of all classes covered by
            all classifiers.
    """
    return np.concatenate(decision_outputs, axis=1)


def transform_label_tensor_to_class_assignment_tensor(label_tensor: np.array,
                                                      n_classes: int) -> np.array:
    """
    Transform a label tensor of shape `(n_classifiers, n_samples)` to the tensor of class assignments of shape
    `(n_classifiers, n_samples, n_classes)`. A label is an integer between `0` and `n_classes - 1`.

    :param label_tensor: `numpy.array` of shape `(n_classifiers, n_samples)`. Label tensor.
    :param n_classes: Number of classes to be considered.
    :return: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`. Class assignment tensor (decision tensor).
    """
    label_tensor = np.array(label_tensor)
    assignments = np.zeros((label_tensor.shape[0], label_tensor.shape[1], n_classes))
    for i in range(label_tensor.shape[0]):
        assignments[i, np.arange(label_tensor.shape[1]), label_tensor[i]] = 1
    return assignments


def transform_label_vector_to_class_assignment_matrix(label_vector: np.array,
                                                      n_classes: int = None) -> np.array:
    """
    Transform labels to multiclass assignments. A vector of shape `(n_samples,)`, with element-wise labels is converted
    to the assignment matrix of shape `(n_samples, n_classes)`.

    :param label_vector: `numpy.array` of shape `(n_samples,)` with an integer label per element.
    :param n_classes: Number of classes to be considered.
    :return: `numpy.array` of shape `(n_samples, n_classes)`. Multiclass assignments.
    """
    label_vector = np.array(label_vector, dtype=int)
    if n_classes is None:
        n_classes = np.max(label_vector) + 1

    assignments = np.zeros((len(label_vector), n_classes), dtype=int)
    assignments[np.arange(len(label_vector)), label_vector] = 1
    return assignments
