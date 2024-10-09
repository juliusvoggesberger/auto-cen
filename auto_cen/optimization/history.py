"""
Module that holds classes that are used to store information of evaluated models.
"""

from collections import namedtuple
from functools import total_ordering


@total_ordering
class RunInfo(
    namedtuple('RunInfo', ['algorithm', 'configuration', 'rnd_seed', 'cost', 'prediction_va',
                           'prediction_ts', 'runtime', 'rid'])):
    """
    Creates a namedtuple.
    ('algorithm': str,  'configuration': dict, 'rnd_seed':int,'cost':float, 'prediction_va':list,
    'prediction_ts':list, 'runtime':float, 'rid':int)
    """
    __slots__ = ()

    def __eq__(self, other):
        return self.algorithm == other.algorithm and self.configuration == other.configuration

    # This could lead to ambiguity but is necessary
    # As it is unknown which cost is the "correct" one, just take the first one
    def __lt__(self, other):
        key = next(iter(self.cost))
        return self.cost[key] < other.cost[key]

    def __hash__(self):
        return hash(self.algorithm) * hash(self.configuration.values())

    def __repr__(self):
        return f"RID: {self.rid};Algorithm: {self.algorithm}; Configuration: {self.configuration};" \
               f" Seed: {self.rnd_seed}; Cost: {self.cost}; Runtime: {self.runtime}"

    def __str__(self):
        return f"RID: {self.rid};Algorithm: {self.algorithm}; Configuration: {self.configuration};" \
               f" Seed: {self.rnd_seed}; Cost: {self.cost}; Runtime: {self.runtime}"

    def __new__(
            cls,  # class type, standard first argument
            algorithm: str,
            configuration: dict,
            rnd_seed: int,
            # Seed used for e.g. generating a bootstrap sample, but NOT for model generation
            cost: dict,  # dict of metrics.
            prediction_va: list,  # Prediction for validation data
            prediction_ts: list,  # Prediction for test data
            runtime: float,
            rid: int,  # Run ID
    ) -> 'RunInfo':
        return super().__new__(cls, algorithm, configuration, rnd_seed, cost, prediction_va,
                               prediction_ts, runtime, rid)


class History:
    """
    A class holding the run history of a solver execution, e.g. the configurations sampled by the
    random search, including the used random seed, the evaluation cost and the runtime (see RunInfo)

    :param runs: A dictionary of runs that has the form:
                 key = algorithm name, values = RunInfos for the algorithm
    """

    def __init__(self, runs=None):
        if runs is None:
            runs = {}
        self.runs: dict = runs

    def add(self, run: RunInfo):
        """
        Adds data of a new run to the history.

        :param run: The information of the run, stored in a RunInfo object.
        """
        if run is None:
            return

        # Add the run
        if run.algorithm in self.runs:
            self.runs[run.algorithm].append(run)
        else:
            self.runs[run.algorithm] = [run]

    def add_history(self, history: 'History'):
        """
        Adds a history of another solver run to the current history
        :param history: The other history
        """
        h_runs = history.get_history()
        if not self.runs:
            # If current history is empty
            self.runs = h_runs
        else:
            for method in h_runs.keys():
                # If method already has runs in the history
                if method in self.runs:
                    self.runs[method] += h_runs[method]
                    self.runs[method] = list(set(self.runs[method]))  # Remove duplicates
                # Else create a new dict entry
                else:
                    self.runs[method] = h_runs[method]

    def get_history(self) -> dict:
        """
        Returns the RunInfos as a dictionary.
        Keys are the algorithm names of the RunInfos, Values a list of the corresponding RunInfos.

        :return: The dictionary.
        """
        return self.runs

    def get_history_list(self) -> list:
        """
        Return all RunInfos of the history object as a list.

        :return: The list.
        """
        runs = [run for algorithm in self.runs.values() for run in algorithm]  # flatten list
        return runs

    def get_algorithm_configs(self, algorithm) -> list:
        """
        Returns a list of the HYPERPARAMETER CONFIGURATIONS of the specified algorithm.

        :param algorithm: The specified algorithm
        :return: A list of hyperparameter configurations of the algorithm.
        """
        runs = self.runs[algorithm]
        configs = [run.configuration for run in runs]
        return configs

    def get_lowest_cost_run(self, metric: str) -> RunInfo:
        """
        Return the RunInfo object with the lowest cost value for a given performance metric.

        :param metric: The performance metric.
        :return: The RunInfo object.
        """
        return min(self.get_history_list(), key=lambda x: x.cost[metric])

    def get_cost_sorted_history(self, metric: str) -> list:
        """
        Returns all RunInfo objects as a list sorted by the cost of a given performance metric.

        :param metric: The performance metric.
        :return: The list.
        """
        # Sorted from low to high cost!
        return sorted(self.get_history_list(), key=lambda x: x.cost[metric])
