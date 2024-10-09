"""
Module that defines an abstract class to be used by every classification orfusion algorithm.
"""

from __future__ import annotations
from abc import ABC
from typing import Type

import numpy as np

from auto_cen.constants import CLASSIFICATION, COMBINATION, SUPPORTED_CLASSIFIER, \
    SUPPORTED_COMBINER
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.utils.checks import classifier_satisfies_specification, \
    combiner_satisfies_specification


class BaseMethod(BaseAlgorithm, ABC):
    """
    Implements a generic abstract class that defines the functions needed for the Methods that are
    part of the optimization problem.

    :param feature_mask: A list of feature indices. Used to select the features for the model.
                        Needed if Random Subspace Method is used.
    :param seed: Random seed.
    """

    # Name abbreviation of the subclass method
    NAME = ""

    def __init__(self, feature_mask: list = None, seed: int = None):
        super().__init__()
        self.feature_mask = feature_mask
        self.seed = seed

    @classmethod
    def get_classifier_for_spec(cls, specification: dict) -> list:
        """
        Given a specification, get all classifier subclasses that satisfy this specification.

        :param specification: A dict with two keys: 'problem' and 'input'.
        :return: A list of classifiers. A classifier is listed as a tuple of its name and its class.
        """
        classifier = []
        for subclass in cls.__subclasses__():
            clf_spec = subclass.get_specification_config()
            if clf_spec['name'] in SUPPORTED_CLASSIFIER and \
                    clf_spec['algorithm'] == CLASSIFICATION \
                    and classifier_satisfies_specification(clf_spec, specification):
                classifier.append((clf_spec['name'], subclass))
        return classifier

    @classmethod
    def get_algorithm(cls, name: str) -> Type[BaseMethod]:
        """
        Given a name, get the algorithm subclass with that name.

        :param name: Name of the algorithm.
        :return: The algorithm class.
        """
        for subclass in cls.__subclasses__():
            clf_spec = subclass.get_specification_config()
            if clf_spec['name'] == name:
                return subclass
        return None

    @classmethod
    def get_combiner_for_spec(cls, specification: dict) -> (list, list):
        """
        Given a specification, get all combiner subclasses that satisfy this specification.

        :param specification: A dict with three keys: 'problem', 'input' and 'output'.
        :return: list of combiners with no hyperparameters, list of combiners with hyperparameters.
        """
        combiner_simple = []
        combiner_hp = []
        for subclass in cls.__subclasses__():
            cmb_spec = subclass.get_specification_config()
            if cmb_spec['name'] in SUPPORTED_COMBINER and cmb_spec['algorithm'] == COMBINATION \
                    and combiner_satisfies_specification(cmb_spec, specification):
                if subclass.get_config_space().estimate_size() == 0.0:
                    combiner_simple.append((cmb_spec['name'], subclass))
                else:
                    combiner_hp.append((cmb_spec['name'], subclass))
        return combiner_simple, combiner_hp

    def predict(self, X: np.array) -> np.array:
        """
        Use the model of the given method to predict labels for given data of shape
        (n_instances, n_features).
        The labels are returned as a vector of shape (n_instances).

        :param X: The data for which the prediction will be made.

        :return: The predicted label as a vector of shape (n_instances).
        """
        raise NotImplementedError

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predict the class probabilities for given data.
        Can only be used by methods that can return class probabilities.

        :param X: The data for which the prediction will be made, of shape (n_instances, n_features)

        :return: The class probabilities for each instance. Has shape (n_instances, n_classes).
        """
        raise NotImplementedError
