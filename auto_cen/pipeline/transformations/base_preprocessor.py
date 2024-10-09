"""
Base module that defines a class which is a template for every preprocessor class.
"""

from __future__ import annotations

from abc import ABC
from typing import Union, Tuple, Type

import numpy as np

from auto_cen.pipeline.base import BaseAlgorithm


class BasePreprocessor(BaseAlgorithm, ABC):
    """
    Implements a generic abstract class that defines the functions needed for all data
    transformation algorithms. This means preprocessing, feature engineering and
    implicit diverisity methods.

    :param seed: Random seed.
    """

    def __init__(self, seed: int = None):
        super().__init__()
        self.seed = seed

    @classmethod
    def get_transformer(cls, name: str) -> Type[BasePreprocessor]:
        """
        Given a transformer name, find its class.

        :param name: The transformer name
        :return: The fitting transformer or None if the name does not exist
        """
        for subclass in cls.__subclasses__():
            trfm = subclass.get_specification_config()
            if trfm['name'] == name:
                return subclass
        return None

    def transform(self, X: np.array, y: np.array = None) -> Tuple[np.array, np.array]:
        """
        Transforms given data.

        :param X: A Data matrix. Optional.
        :param y: A Label array. Optional.
        :return: Either the transformed X or y, or a tuple of the both transformed (X,y)
        """
        raise NotImplementedError
