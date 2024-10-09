from sklearn.metrics import confusion_matrix

from auto_cen.pusion.utils.transformer import *


def generate_multiclass_confusion_matrices(decision_tensor: np.array,
                                           true_assignments: np.array) -> np.array:
    """
    Generate multiclass confusion matrices out of the given decision tensor and true assignments.
    Continuous outputs are converted to multiclass assignments using the MAX rule.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different classifiers per sample.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp class assignments which are considered true for calculating confusion matrices.
    :return: `numpy.array` of shape `(n_classifiers, n_classes, n_classes)`. Confusion matrices per classifier.
    """
    true_assignment_labels = np.argmax(true_assignments, axis=1)
    confusion_matrices = np.zeros((np.shape(decision_tensor)[0],
                                   np.shape(true_assignments)[1],
                                   np.shape(true_assignments)[1]), dtype=int)
    for i in range(len(decision_tensor)):
        decision_tensor_labels = np.argmax(decision_tensor[i], axis=1)
        confusion_matrices[i] = confusion_matrix(y_true=true_assignment_labels,
                                                 y_pred=decision_tensor_labels,
                                                 labels=np.arange(np.shape(true_assignments)[1]))
    return confusion_matrices
