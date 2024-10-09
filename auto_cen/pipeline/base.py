"""
Base module holding an abstract class that is a template for all class in the machine learning
pipeline.
"""

from __future__ import annotations
from abc import ABC

import numpy as np
from ConfigSpace import ConfigurationSpace


class BaseAlgorithm(ABC):
    """
    Implements a generic abstract class that defines the functions needed for all Algorithms
    that are used in a Machine Learning pipeline. This includes preprocessing, feature engineering
    and implicit diversity methods, as well as classification and fusion algorithms.
    """

    def __init__(self):
        pass

    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:
        """
        Fit the model of the given algorithm.
        If the algorithm does not need to be trained, return nothing.

        :param X: The data needed to fit the algorithm.
        :param y: The labels needed to train the algorithm. Optional
        """
        raise NotImplementedError

    def get_params(self, deep=True) -> dict:
        """
        Returns the configuration of the algorithm.
        It includes the specification of the algorithm and its hyperparameters with set values.

        :param deep:
        :return: A dictionary including the specification and hyperparameters.
        """
        raise NotImplementedError

    @staticmethod
    def get_specification_config() -> dict:
        """
        Returns the specification of the algorithm.
        This includes the name, algorithm type, etc..

        :return: A dictionary of the specification. Keys are:
                 algorithm type, algorithm name, is_deterministic, input, problem and output.
        """
        raise NotImplementedError

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        """
        Returns the configuration space of the algorithm as a SMAC config space representation.

        :return: The configuration space of the algorithm.
                 If the method has no hyperparameters, `None` is returned.
        """
        raise NotImplementedError


class BasePipeline(BaseAlgorithm, ABC):
    """
    Implements a generic abstract class for implementing pipeline classes.
    Inherits from 'BaseAlgorithm' and adds the methods 'predict' and 'predict_proba'.

    :param algorithm: The algorithm name as a string.
                      Should contain an algorithm as defined in constants.py, either in:
                      SUPPORTED_CLASSIFIER or SUPPORTED_COMBINER.
    :param configuration: The configuration of the algorithm and possible preprocessing steps.
    :param seed: Random seed.
    """

    def __init__(self, algorithm: str, configuration: {}, seed: int = None):
        super().__init__()
        self.algorithm = algorithm
        self.raw_configuration = configuration
        self.seed = seed

    def predict(self, X: np.array) -> np.array:
        """
        Predict labels for a given dataset.

        :param X: The data set.

        :return: The predicted labels.
        """
        raise NotImplementedError

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predict probabilities per label for a given dataset.

        :param X: The data set.

        :return: The predicted probabilities for each label.
        """
        raise NotImplementedError
