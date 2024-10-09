"""
Implements a wrapper for the pusion Weighted Voting implementation.
"""

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter

from auto_cen.constants import COMBINATION, MULTICLASS, LABELS, CONTINUOUS_OUT, EVIDENCE_COMBINER
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pusion import WeightedVotingCombiner


class WeightedVoting(BaseMethod):
    """
    Wrapper for the pusion combiner Weighted Voting.

    :param scorer: Scoring function used for computing the weights.
                   Can be 'accuracy', 'macro_recall', 'micro_recall', 'macro_precision',
                   'micro_precision', 'macro_f1' or 'micro_f1'
    :param weighting: Weighting function used for transforming the scores.
                      Can be 'simple', 'rescaled', 'best_worst', 'quadratic', 'logarithmic'.
    """

    def __init__(self, scorer: str, weighting: str, **kwargs):
        super().__init__(feature_mask=kwargs["feature_mask"], seed=kwargs["seed"])

        self.scorer = scorer
        self.weighting = weighting

        # Pass the functions instead of the hyperparamters to avoid changing too much code in pusion
        self.model = None

    def fit(self, X: list, y: list):
        n_classes = len(y[0])
        self.model = WeightedVotingCombiner(score_f=self.scorer, weight_f=self.weighting,
                                            n_classes=n_classes)
        self.model.train(X, y)

    def predict(self, X: list) -> list:
        return self.model.combine(X)

    def predict_proba(self, X: list) -> list:
        pass

    def get_params(self, deep=True) -> dict:
        return {'scorer': self.scorer,
                'weighting': self.weighting,
                'feature_mask': self.feature_mask,
                'seed': self.seed, }

    @staticmethod
    def get_specification_config() -> dict:
        return {'name': 'WV',
                'algorithm': COMBINATION,
                'is_deterministic': True,
                'combiner_type': EVIDENCE_COMBINER,
                'input': (LABELS, CONTINUOUS_OUT),
                'problem': (MULTICLASS,),
                'output': (LABELS,)}

    @staticmethod
    def get_config_space() -> ConfigurationSpace:
        c_space = ConfigurationSpace()

        scorer_func = CategoricalHyperparameter('scorer',
                                                ['accuracy', 'macro_recall', 'micro_recall',
                                                 'macro_precision', 'micro_precision',
                                                 'macro_f1', 'micro_f1'],
                                                default_value='macro_recall')
        weight_func = CategoricalHyperparameter('weighting',
                                                ['simple', 'rescaled', 'bw', 'bw_quadratic',
                                                 'logarithmic'], default_value='bw')

        c_space.add_hyperparameters([scorer_func, weight_func])

        return c_space
