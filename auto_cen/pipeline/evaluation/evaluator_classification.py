"""
An Evaluator class, which is used to evaluate a given classifier algorithm and its hyperparameter
configuration.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from auto_cen.main.data_storage import DataStore
from auto_cen.pipeline.evaluation.base_evaluator import BaseEvaluator
from auto_cen.pipeline.generic_clf_pipeline import ClassifierPipeline
from auto_cen.utils.performance_metrics import get_cost_report


class ClassificationEvaluator(BaseEvaluator):
    """
    A class which is used to evaluate classifier methods.
    The given classifier and configuration is evaluated using K-Fold cross-validation on the
    training data. A model is then trained using the given training data.
    The trained model is lastly used to compute predictions for the given validation and test data.

    :param data: DataStore object containing all data.
    :param metrics: List of performance metrics to use in the evaluation.
    :param sel_metric: The performance metric that will be used to compute the cost.
    :param n_splits: Number of cross validation splits.
    """

    def __init__(self, data: DataStore, metrics: list, sel_metric: str, n_splits: int = 1):
        super().__init__(metrics, sel_metric, n_splits)
        self.data = data

    def evaluate(self, algorithm: str, configuration: dict, seed: int = None) -> (list, dict):
        """
        Evaluate a given algorithm and configuration.
        The evaluation score is computed via cross-validation.

        :param algorithm: The algorithm name.
        :param configuration: The hyperparameter configuration of the algorithm.
        :param seed: Random seed.
        :return: The predictions of the evaluation data, the cost(s).
        """
        cost = None
        if self.n_splits >= 2:
            # Cross validation will be used, as such the config has to be retrained later on
            cost, preds = self._cv_multiple_metrics(algorithm, configuration, seed)

        # (Re)train the model and compute predictions, needed later on for the decision fusion
        cp = ClassifierPipeline(algorithm, configuration, seed, self.data.cat_indices)
        cp.fit(self.data.X_train, self.data.y_train)
        y_predv = cp.predict(self.data.X_valid) if self.n_splits < 2 else preds
        y_predt = cp.predict(self.data.X_test) if self.n_splits < 2 else np.array([])

        if cost is None:  # If cv was used, cost was already computed
            cost = get_cost_report(self.data.y_valid, y_predv, None, self.metrics,
                                   self.data.n_classes)
        return [y_predv.tolist(), y_predt.tolist()], cost

    def get_selection_metric(self):
        return self.sel_metric

    def _cv_multiple_metrics(self, algorithm: str, configuration: dict, seed: int) -> (dict, np.array):
        """
        Computes cross-validation for performance metrics.

        :param algorithm: The algorithm name.
        :param configuration: The hyperparameter configuration of the algorithm.
        :param seed: Random seed.

        :return: dict of scores and predictions of all estimators.
        """
        cp = ClassifierPipeline(algorithm, configuration, seed, self.data.cat_indices)
        kfold = StratifiedKFold(n_splits=self.n_splits)

        cv_results = cross_validate(cp, self.data.X_train, self.data.y_train,
                                    cv=kfold.split(self.data.X_train, self.data.y_train),
                                    scoring="balanced_accuracy", error_score='raise',
                                    return_estimator=True, return_indices=True)
        costs = cv_results['test_score']

        # Get the predictions
        predictions = np.zeros(len(self.data.y_train), dtype=np.int8)
        for i in range(self.n_splits):
            test = cv_results['indices']["test"][i]
            predictions[test] = cv_results['estimator'][i].predict(self.data.X_train[test])
        costs = {self.sel_metric: 1 - np.mean(costs)}

        return costs, predictions
