"""
Module containing an abstract class which has to be used by every class that is used to optimize
the CASH problem.
"""

from abc import ABC, abstractmethod

from ConfigSpace import ConfigurationSpace

from auto_cen.optimization.history import History
from auto_cen.pipeline.evaluation.base_evaluator import BaseEvaluator


class Solver(ABC):
    """
    Provides an abstract interface to implement hyperparameter optimization solvers.
    """

    NAME = ""

    def __init__(self, c_space: ConfigurationSpace, budget: int, evaluator: BaseEvaluator,
                 simplified_c_space: list = None, seed: int = None):
        """
        Abstract class for a hyperparameter optimization solver.

        :param c_space: Configuration Space which will be searched by the solver.
        :param budget: The budget is the time used for the optimization.
        :param evaluator: Evaluator class used to evaluate a sampled configuration.
        :param simplified_c_space: If not None, a list that contains algorithms that have no
                              hyperparameters and will be trained before searching the
                              Configuration Space 'cs'
        :param seed: Random seed.
        """

        self.c_space = c_space
        self.simplified_cs = simplified_c_space
        self.budget = budget
        self.evaluator = evaluator

        self.seed = seed

    @abstractmethod
    def run(self, cutoff_time: int = 1800, n_processes: int = 4) -> History:
        """
        Run the solver for given data.

        :param cutoff_time: The time (in sec) after which a configuration training will be stopped.
        :param n_processes: Number of processes to be used, if the solver is run in parallel.
                            Default 4.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_history(self):
        """
        Return the history object of the solver.
        Contains the runs evaluated by the solver.
        """
        raise NotImplementedError()
