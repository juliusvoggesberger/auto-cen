"""
Implements a wrapper for the pusion Decision Template implementation.
"""

import numpy as np

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter

from auto_cen.constants import COMBINATION, MULTILABEL, MULTICLASS, LABELS, CONTINUOUS_OUT, \
    TRAINABLE_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import DecisionTemplatesCombiner


class DecisionTemplate(BaseMethod):
    """
    Wrapper for the pusion combiner Decision Template.

    :param similarity: The similarity metric used. Can be 'euclidean', 'manhattan', 'cosine' or
                       'symmetric_diff'
    """

    def __init__(self, similarity, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.similarity = similarity

        self.model = DecisionTemplatesCombiner(similarity=self.similarity)

    def fit(self, X: np.array, y: np.array):
        self.model.train(X, y)

    def predict(self, X: np.array) -> np.array:
        return self.model.combine(X)

    def predict_proba(self, X: np.array) -> np.array:
        raise NotImplementedError('Can\'t predict class probabilities!')

    def get_params(self, deep=True) -> dict:
        return {
            'similarity': self.similarity,
            'feature_mask': self.feature_mask,
            'seed': self.seed
        }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'DTEMP',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': TRAINABLE_COMBINER,
                'input': (LABELS, CONTINUOUS_OUT),
                'problem': (MULTICLASS, MULTILABEL),
                'output': (LABELS,)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        similarity = CategoricalHyperparameter('similarity', ['euclidean', 'manhattan', 'cosine',
                                                              'symmetric_diff'],
                                               default_value='euclidean')

        c_space.add_hyperparameter(similarity)

        return c_space
