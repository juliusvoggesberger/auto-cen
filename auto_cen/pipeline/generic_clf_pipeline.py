"""
Module for executing a complete classifier pipeline, including pre-processing, feature-engineering,
implicit diversity and the classifier itself.
"""

import numpy as np
import scipy.sparse

from ConfigSpace import ConfigurationSpace

from auto_cen.constants import MIXED, PIPELINE, LABELS, CONTINUOUS_OUT, ENCODER
from auto_cen.pipeline.base import BasePipeline
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pipeline.transformations.base_preprocessor import BasePreprocessor
from auto_cen.utils.utils import process_configuration


class ClassifierPipeline(BasePipeline):
    """
    Class for executing a complete classifier pipeline, including pre-processing,
    feature-engineering, implicit diversity and the classifier itself.

    The different steps of the pipeline are given as a configuration dictionary and the classifier
    algorithm is explicitly stated as a string.

    :param clf_algorithm: The classification algorithm.
    :param configuration: The configuration of the pipeline, including the hyperparameters of the
                          classifier.
    :param seed: Random seed.
    """

    def __init__(self, clf_algorithm: str, configuration: dict, seed: int, categories: list):
        super().__init__(clf_algorithm, configuration, seed)

        self.configurations = process_configuration(['Enc', 'Pre', 'Fe', 'Div', self.algorithm],
                                                    self.algorithm, self.raw_configuration)

        self.categories = categories

        self.pre_steps = []
        self.model = None
        self.fm = None

    def fit(self, X: np.array, y: np.array) -> 'ClassifierPipeline':
        """
        Fits the pipeline.

        :param X: Training data
        :param y: Training labels

        :return: self object
        """

        # Needed for implicit diversity
        for k, v in self.configurations.items():

            alg, config = v
            if alg == ENCODER:
                config["categories"] = self.categories
            transf = BasePreprocessor.get_transformer(alg)

            if alg != self.algorithm and transf is not None:

                transf = transf(**config, seed=self.seed).fit(X, y)
                X, y = transf.transform(X, y)

                if k != 'Div':
                    self.pre_steps.append(transf)
                if hasattr(transf, 'feature_mask'):
                    self.fm = transf.feature_mask
                if scipy.sparse.issparse(X):
                    # TODO support sparse arrays natively
                    X = X.toarray()
        _, config = self.configurations.get(self.algorithm)
        self.model = BaseMethod.get_algorithm(self.algorithm)(**config, feature_mask=self.fm,
                                                              seed=self.seed)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predict labels for given dataset.

        :param X: The data set.

        :return: The predicted labels.
        """
        X = self._apply_transformations(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.array) -> np.array:
        """
        Predict labels for a given dataset.

        :param X: The data set.

        :return: The prediction probabilities for all labels.
        """

        X = self._apply_transformations(X)
        return self.model.predict_proba(X)

    def get_params(self, deep=True) -> dict:
        return {
            'clf_algorithm': self.algorithm,
            'configuration': self.raw_configuration,
            'seed': self.seed,
            'categories': self.categories
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'PIPE',
                'algorithm': PIPELINE,
                'is_deterministic': True,
                'input': MIXED,
                'problem': (PIPELINE,),
                'output': (LABELS, CONTINUOUS_OUT)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()
        return c_space

    def _apply_transformations(self, X):
        for step in self.pre_steps:
            if step is not None:
                X, _ = step.transform(X)
            if scipy.sparse.issparse(X):
                # TODO support sparse arrays natively
                X = X.toarray()
        return X
