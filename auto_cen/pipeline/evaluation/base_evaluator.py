"""
Base class for algorithm configuration evaluators.
"""

from abc import ABC
from typing import Type

from auto_cen.constants import ACCURACY
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod


class BaseEvaluator(ABC):
    """
    An abstract class, that defines the two methods that need to be implemented by an evaluator.
    An evaluator is used to evaluate a given algorithm and configuration.

    :param metrics: Performance metrics that are computed.
    :param sel_metric: The performance metric that will be used to compute the cost.
    :param n_splits: Number of cross validation splits.
    """

    def __init__(self, metrics=None, sel_metric: str = ACCURACY, n_splits: int = 1):
        if metrics is None:
            metrics = [ACCURACY]
        self.metrics = metrics
        self.sel_metric = sel_metric
        self.n_splits = n_splits

    def evaluate(self, algorithm: str, configuration: dict, seed: int = None) -> (list, dict):
        """
        Evaluate a given algorithm and configuration.

        :param algorithm: The algorithm name.
        :param configuration: The hyperparameter configuration of the algorithm.
        :param seed: Random seed.
        :return: The prediction(s), the difference between the prediction and the true labels,
                 a dictionary containing the cores of each metric
        """
        raise NotImplementedError

    def get_selection_metric(self) -> str:
        """
        Returns the metric which will be used to rate the algorithms.
        """
        return self.sel_metric
