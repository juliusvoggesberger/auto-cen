"""
Module holding the bayesian optimization class used for optimizing the CASH problem.
This class is basically a wrapper for smac3.
"""

import random

import numpy as np

from ConfigSpace import Configuration, ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
from smac.tae import StatusType

from auto_cen.optimization.history import History, RunInfo
from auto_cen.optimization.solver import Solver
from auto_cen.pipeline.evaluation.base_evaluator import BaseEvaluator


class BayesianOptimization(Solver):
    """
    Implements Random Search for solving the CASH-Problem.
    Given a configuration space, a time budget and an evaluator, the solver searches for
    configurations.

    :param c_space: The configuration space as a list.
    :param budget: A time budget, if the time runs up the last configuration sample is finished and
                   then the solver is stopped.
    :param evaluator: Evaluator class used to evaluate a sampled configuration.
    :param simple_cs: Not supported for Bayesian optimization. Will not do anything.
    :param seed: Random seed.
    """

    NAME = "BO"

    def __init__(self, c_space: ConfigurationSpace, budget: int, evaluator: BaseEvaluator,
                 simple_cs=None, seed: int = None):
        super().__init__(c_space, budget, evaluator, simple_cs, seed)
        self.configurations = History()
        self.seed = seed

    def run(self, cutoff_time: int = 1800, n_processes: int = 4) -> History:
        """
        Runs the solver.

        :return: A History object, containing the Run Information of the different Configuration
                 evaluations
        """
        scenario = Scenario({
            'abort_on_first_run_crash': False,
            'run_obj': 'quality',
            'wallclock-limit': self.budget,
            'cutoff': cutoff_time,
            'cs': self.c_space,
            'deterministic': True,
            'shared_model': True,  # Needed to run in parallel
            'output_dir': "",
            'input_psmac_dirs': '/psmac_shared',
            # Needed to run in parallel
            'limit_resources': True,  # This is necessary, so that the cutoff time can be applied
            # 'runcount-limit': round(self.budget / n_processes)
        })

        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
        else:
            rng = None

        # The tae_runner is expected to be either of type BaseRunner or some other type (see doc)
        # according to typehints But passing a function also works, as for the other
        # - non-parallel - facades
        if n_processes == 1:
            smac = SMAC4AC(scenario=scenario, rng=rng, tae_runner=self._tae_wrapper)
        else:
            from smac.facade.psmac_facade import PSMAC
            # noinspection PyTypeChecker
            smac = PSMAC(scenario=scenario, rng=rng, tae_runner=self._tae_wrapper,
                         n_workers=n_processes, shared_model=True)
        smac.optimize()

        hist = smac.get_runhistory()
        for x, y in hist.data.items():
            self._smac_to_native_run(x, y, hist.ids_config)
        return self.configurations

    def get_history(self):
        return self.configurations

    def _tae_wrapper(self, config: Configuration, seed: int) -> (float, list):
        """
        A wrapper for the given evaluator to conform to SMACs Target Algorithm Evaluator (TAE)
        function requirements. The return value has to be a float value, which is the loss-value
        for the evaluated algorithm.
        Important: SMAC is optimizing by minimizing. As such performance metrics like accuracy have
                   to be altered accordingly, i.e. return 1-accuracy.

        Normally also takes the following parameters:
            - seed: Already set in the Evaluator
            - instance: Data which to use to train the model. For more info on how this is done see:
                        https://automl.github.io/SMAC3/master/pages/details/instances.html
                        and as an example:
                        https://automl.github.io/SMAC3/main/examples/python/plot_sgd_instances.html


        :param config: Configuration of the target algorithm.
        :param seed: Random seed.
        :return: The evaluation metric, A dictionary containing additional information
        """
        y_pred_va, y_pred_ts = [], []
        if seed == 0.0:
            # If no seed is set by the user - generate one here, as it is needed to replicate
            # implicit diversity when retraining the configurations later on
            if self.seed is not None:
                seed = self.seed
            else:
                seed = random.randrange(0, 2147483647)

        config_dict = dict(config).copy()
        algorithm = config_dict.pop('algorithm')
        try:
            y_pred, cost = self.evaluator.evaluate(algorithm, config_dict, seed)
        except np.linalg.LinAlgError or ValueError:
            # This is done to prevent spamming the log full of LinAlgErrors from LDA crashes (LinAlg)
            # This is done to prevent spamming the log full of ValueErrors from MLP crashes (Value)
            return 2147483647.0, {'algorithm': StatusType.CRASHED}

        eval_metric = self.evaluator.get_selection_metric()
        if y_pred is not None:
            y_pred_va, y_pred_ts = y_pred[0], y_pred[1]

        add_info = {
            'algorithm': algorithm,
            'prediction_va': y_pred_va,
            'prediction_ts': y_pred_ts,
            'cost': self._make_json_serializable(cost),
            'sample_seed': seed,
        }

        return cost[eval_metric], add_info

    def _smac_to_native_run(self, run_item1: tuple, run_item2: tuple, history_id_config: dict):
        """
        Converts a Run Information object of SMAC to a native RunInfo object.
        See the SMAC Documentation, or the example at:
        https://automl.github.io/SMAC3/main/details/run_history.html

        :param run_item1: The basic information of the run
        :param run_item2: The additional information of the run
        :param history_id_config: The configuration dictionary. Maps a config id
                                  (saved in run_item1) to a configuration
        """
        config_id = run_item1[0]
        time = run_item2[1]
        status = run_item2[2]
        run_info = run_item2[5]

        if status != StatusType.SUCCESS or run_info['algorithm'] == StatusType.CRASHED:
            # If StatusType is not SUCCESS something went wrong
            return

        run = RunInfo(algorithm=run_info['algorithm'],
                      configuration=history_id_config[config_id].get_dictionary(),
                      rnd_seed=run_info['sample_seed'],
                      cost=self._reconvert_to_numpy(run_info['cost']),
                      prediction_va=run_info['prediction_va'],
                      prediction_ts=run_info['prediction_ts'],
                      runtime=time,
                      rid=config_id)

        self.configurations.add(run)

    @staticmethod
    def _make_json_serializable(cost: dict):
        """
        Transforms numpy arrays to python lists, as SMAC serializes the run information with json.

        :param cost: The cost dictionary.
        :return: The json serializable dictionary.
        """
        for k, v in cost.items():
            if isinstance(v, np.ndarray):
                cost[k] = v.tolist()
        return cost

    @staticmethod
    def _reconvert_to_numpy(cost: dict):
        """
        After running SMAC the numpy -> list transformation should be reversed.
        :param cost: The cost dictionary.
        :return: The transformed cost dictionary.
        """

        for k, v in cost.items():
            if isinstance(v, list):
                cost[k] = np.array(v)
        return cost
