"""
An Evaluator class, which is used to evaluate a given fusion algorithm and its hyperparameter
configuration.
"""


def warn(*args, **kwargs):
    pass


import warnings
from typing import List

import numpy as np
from sklearn.model_selection import StratifiedKFold

from auto_cen.constants import ACCURACY
from auto_cen.main.data_storage import DataStore
from auto_cen.pipeline.base import BaseAlgorithm

from auto_cen.pipeline.evaluation.base_evaluator import BaseEvaluator
from auto_cen.pipeline.generic_cmb_pipeline import CombinerPipeline
from auto_cen.pipeline.selector import Selector

from auto_cen.pusion.utils import multiclass_assignments_to_labels
from auto_cen.utils.performance_metrics import get_performance_report, get_cost_report

warnings.warn = warn  # TODO Not nice, but sklearn forces the User Warning

class CombinerEvaluator(BaseEvaluator):
    """
    A class which is used to evaluate combiner methods.
    The given combiner with configuration is evaluated using cross-validation on the training data,
    i.e. the prediction of the ensemble for the validation data (y_valid_ensemble_prob, y_valid).
    The combiner is then trained on that data.

    :param data: Fusion Data containing the training, valid and true data for the fusion methods.
    :param metrics: List of performance metrics to use in the evaluation.
    :param sel_metric: The performance metric that will be used to compute the cost.
    :param n_splits: Number of cross validation splits.
    """

    def __init__(self, data: DataStore, metrics: list = None, sel_metric: str = ACCURACY,
                 n_splits: int = 1, selector: Selector = None):
        super().__init__(metrics, sel_metric, n_splits)
        self.data = data
        self.selector = selector

        self.models = None

    def evaluate(self, algorithm: str, configuration: dict, seed: int = None) -> (list, dict):
        """
        Evaluate a given algorithm and configuration.

        :param algorithm: The algorithm name.
        :param configuration: The hyperparameter configuration of the algorithm.
        :param seed: Random seed.
        :return: None, the evaluation score
        """

        combiner = CombinerPipeline(algorithm, configuration, self.selector, seed)
        if self.models is not None:
            combiner.set_classifiers(self.models)

        # Use predictions of the models on the validation data to train
        X_train, y_train, X_test, y_test = self.data.compute_fusion_data(combiner.classifiers)

        if self.n_splits >= 2:
            # Compute costs, if cross-validation
            cost = self._cross_validation(algorithm, configuration, X_train, y_train, seed)
        else:
            # Compute costs, if not cross-validation
            combiner.fit(X_train, y_train)
            y_pred = np.asarray(combiner.predict(X_test))
            cost = get_cost_report(y_test, y_pred, None, self.metrics, self.data.n_classes)

        return None, cost

    def get_selection_metric(self):
        return self.sel_metric

    def _cross_validation(self, algorithm: str, configuration: dict, X: np.array,
                          y_label_tensor: np.array, seed: int = None) -> dict:
        """
        Implements k-fold cross validation for combiner algorithms.
        The vanilla sklearn method can not be used, as the first dimension of the shape of the
        training data (n_samples, n_classifier, n_classes) does not fit to the dimension of the
        label data (n_samples,) or (n_samples, n_labels) if multilabel.

        :param algorithm: The combiner algorithm.
        :param configuration: The configuration.
        :param X:  The X training data.
        :param y_label_tensor: The labels as a tensor.
        :param seed: Random seed.
        :return: The validation score
        """
        kfold = StratifiedKFold(n_splits=self.n_splits)
        scores = {m: [] for m in self.metrics}
        # Swap axes, as training data has shape (n_classifier, n_samples, n_classes) and labels
        # have shape (n_samples,) or (n_samples, n_labels) if multilabel.
        # To use KFold a shape of (n_samples, n_classifier, n_classes) is needed instead, as the
        # first dimensions have to fit with the labels.
        train_swap = np.swapaxes(X, 0, 1)
        # Labels are needed for stratified k-fold
        y_labels = multiclass_assignments_to_labels(y_label_tensor)
        for train, test in kfold.split(train_swap, y_labels):
            # Data, swap axes of X back
            # X_train will have the subset (n_classifiers, n_samples[train], n_classes)
            # X_test will have the subset (n_classifiers, n_samples[test], n_classes)
            X_train, y_train = np.swapaxes(train_swap[train], 0, 1), y_label_tensor[train]
            X_test, y_test = np.swapaxes(train_swap[test], 0, 1), y_label_tensor[test]

            # Fit predict
            model = CombinerPipeline(algorithm, configuration, self.selector, seed)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            costs = get_performance_report(y_test, y_pred, None, self.metrics, self.data.n_classes,
                                           labels=list(range(self.data.n_classes)))

            for k, cost in costs.items():
                scores[k].append(cost)

        # Compute the mean for each metric and get 1-score to get the cost
        scores = {k: 1 - np.mean(v, axis=0) for k, v in scores.items()}
        return scores

    def set_classifiers(self, models: List[BaseAlgorithm]):
        self.models = models
