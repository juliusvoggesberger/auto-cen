"""
Module for executing a complete combiner pipeline, including extracting the classifier models, if
needed.
"""

import numpy as np
from ConfigSpace import ConfigurationSpace

from auto_cen.constants import PIPELINE, MIXED, LABELS, CONTINUOUS_OUT
from auto_cen.pipeline.base import BasePipeline
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pipeline.selector import Selector
from auto_cen.utils.utils import process_configuration


class CombinerPipeline(BasePipeline):
    """
    Class for executing a complete classifier pipeline, including pre-processing,
    feature-engineering, implicit diversity and the classifier itself.

    The different steps of the pipeline are given as a configuration dictionary and the classifier
    algorithm is explicitly stated as a string.

    :param cmb_algorithm: The combiner/fusion algorithm.
    :param configuration: The configuration of the pipeline, including the hyperparameters of the
                          combiner and the ensemble size.
    :param seed: Random seed.
    """

    def __init__(self, cmb_algorithm: str, configuration: dict, selector: Selector = None,
                 seed: int = None):
        super().__init__(cmb_algorithm, configuration, seed)

        self.configurations = process_configuration(['ENS', cmb_algorithm], cmb_algorithm,
                                                    configuration)

        self.model = BaseMethod.get_algorithm(self.algorithm)(
            **self.configurations.get(self.algorithm)[1], seed=self.seed, feature_mask=None)
        self.classifiers = None

        self.ens_size = self.configurations["ENS"][1]["size"]
        if selector is not None:
            self.classifiers = selector.select(self.ens_size)

    def fit(self, X: np.array, y: np.array) -> BasePipeline:
        # Fit the combiner/fusion method to the given data
        self.model.fit(X, y)
        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predict labels for given dataset.

        :param X: The data set.

        :return: The predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.array) -> np.array:
        raise NotImplementedError('Can not predict class probabilities!')

    def get_params(self, deep=True) -> dict:
        return {
            'cmb_algorithm': self.algorithm,
            'configuration': self.raw_configuration,
            'seed': self.seed,
        }

    def get_model(self):
        return self.model

    def set_classifiers(self, classifiers: list):
        self.classifiers = classifiers

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'PIPE_CMB',
                'algorithm': PIPELINE,
                'is_deterministic': True,
                'input': MIXED,
                'problem': (PIPELINE,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        return c_space
