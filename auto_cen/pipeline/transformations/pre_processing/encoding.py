"""
Module implementing a wrapper for categorical data encoding.
"""

import numpy as np
from ConfigSpace import ConfigurationSpace, Constant
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from auto_cen.constants import MIXED, ENCODER, PREPROCESSOR
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor


class Encoder(BasePreprocessor):
    """
    Encodes categorical data as integers.

    :param encoder: Either "ordinal" or "one-hot" depending on the encoder to be used.
    :param seed: Random seed.
    :return: The rescaled data.
    """

    def __init__(self, encoder: str, seed: int = None, categories: list = "auto"):
        super().__init__(seed)
        self.encoder = encoder
        if self.encoder == "ordinal":
            model = OrdinalEncoder(handle_unknown="ignore")
        elif self.encoder == "one-hot":
            model = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.model = ColumnTransformer(transformers=[("cat", model, categories)], remainder="passthrough")

    def fit(self, X: np.array, y: np.array) -> BaseAlgorithm:
        self.model.fit(X)
        return self

    def transform(self, X: np.array = None, y: np.array = None) -> (np.array, np.array):
        X_enc = self.model.transform(X)
        return X_enc, y

    def get_params(self, deep=True) -> dict:
        return {
            'encoder': self.encoder,
            'seed': self.seed,
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': ENCODER,
                'algorithm': PREPROCESSOR,
                'is_deterministic': True,
                'input': MIXED,
                }

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        # encoder = CategoricalHyperparameter('encoder', ["ordinal", "one-hot"])
        encoder = Constant('encoder', value= "one-hot")
        c_space.add_hyperparameters([encoder])

        return c_space