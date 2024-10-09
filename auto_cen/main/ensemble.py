"""
Main module that calls and executes all relevant functions and modules.
"""
import pickle
import time
import logging
from typing import Union, Type, Tuple

import pandas as pd
import numpy as np
from ConfigSpace import ConfigurationSpace

from auto_cen.inout.EnsembleAnalysis import EnsembleAnalysis
from auto_cen.utils.input_validation import generate_specification, get_combiner

from auto_cen.main.data_storage import DataStore
from auto_cen.main.utils import retrain_ensemble
from auto_cen.main.cs_creator import create_cs

from auto_cen.optimization.bo import BayesianOptimization
from auto_cen.optimization.history import RunInfo, History
from auto_cen.optimization.solver import Solver

from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod

from auto_cen.pipeline.evaluation.base_evaluator import BaseEvaluator
from auto_cen.pipeline.evaluation.evaluator_fusion import CombinerEvaluator
from auto_cen.pipeline.evaluation.evaluator_classification import ClassificationEvaluator
from auto_cen.pipeline.selector import ClusterSelection, Selector

from auto_cen.pusion.utils.transformer import transform_label_tensor_to_class_assignment_tensor, \
    multiclass_assignments_to_labels
from auto_cen.constants import MULTILABEL, ACCURACY, DOUBLEFAULT, CV, BOOTSTRAP, RSM, PS, NOISE, \
    FLIP, SILHOUETTE

logger = logging.getLogger('auto_cen')


class EnsembleLearner:
    """
    The Ensemble Learner class.
    Holds the information for automatically training an ensemble learner.

    :param ens_size: The size of the ensemble. Can be either an integer or a tuple.
                     If a tuple, this will be the interval in which the ensemble size will be
                     optimized.
    :param budget_m: The number of ML-configurations that have to be evaluated.
    :param budget_f: The number of Combiner-configurations that have to be evaluated.
    :param solver: The class which to use as a HPO Solver. Default is Bayesian Optimization.
    :param perf_metric: A metric to measure the performance. Default: accuracy.
    :param div_metric: A metric to measure the diversity. Default: double-fault.
    :param eval_perf_metrics: List of performance metrics used to evaluate.
           If None, then just the given performance metric is used.
           Using multiple metrics can affect performance.
    :param eval_div_metrics: List of diversity metrics used to evaluate.
                             Using multiple metrics can affect performance.
    :param sel_heuristic: EXPERIMENTAL!
                          If "Diversity" uses non-pairwise diversity to select a set of the
                          x% models (x=10/20/../100)
                          If "Performance" selects the best performing 50% of models.
                          If "Averaged" uses the average method from caruana et al.
                          If "OrientationOrdering" uses orientation ordering.
                          If "SIL" uses Silhouette to compute the optimal number of classifiers
                          If None: Uses all models.
    :param find_ensemble_size: True, if all 2 .. ensemble_size, sizes should be tried.
                               For each size budget_f will be used.
    :param n_splits: Number of splits used for the cross-validation.
                     Default 1, i.e. no cv will be used.
                     If > 1, cv will be used. As such only two data sets (Train, Validation) will
                     have to be passed.
    :param cutoff_time: The time (in seconds) after which a configuration training will be stopped
                        by BO.
    :param seed: A random state which will be used - if set to an integer - ,
                 to make the results deterministic.
    """

    def __init__(
            self,
            ens_size: Union[int, Tuple],
            budget_m: int,
            budget_f: int,
            solver: Type[Solver] = BayesianOptimization,

            perf_metric: str = ACCURACY,
            div_metric: str = DOUBLEFAULT,
            eval_perf_metrics: list = None,
            eval_div_metrics: list = None,

            sel_heuristic: str = None,
            find_ensemble_size: bool = False,
            n_splits: int = 1,

            cutoff_time: int = 1800,
            seed: int = None
    ):

        # if isinstance(ens_size, int) and ens_size < 2:
        #    raise ValueError("Ensemble Size should be larger then 1")

        if eval_perf_metrics is None:
            eval_perf_metrics = [perf_metric]
        if perf_metric not in eval_perf_metrics:
            eval_perf_metrics.append(perf_metric)

        if eval_div_metrics is None:
            eval_div_metrics = [div_metric]
        if div_metric not in eval_div_metrics:
            eval_div_metrics.append(div_metric)

        # Needed user input
        self.ensemble_size, self.budget_m, self.budget_f = ens_size, budget_m, budget_f
        self.cutoff_time = cutoff_time
        self.sel_heuristic = sel_heuristic

        # Optional user input
        self.solver = solver
        self.find_ensemble_size = find_ensemble_size
        self.seed, self.n_splits = seed, n_splits
        self.perf_metric, self.div_metric = perf_metric, div_metric
        self.eval_perf_metrics, self.eval_div_metrics = eval_perf_metrics, eval_div_metrics

        self.specification = {}
        self.combiner_hp, self.combiner_simple = [], []
        self.classifiers = []

        self.data = None

        # Selected Ensemble
        self.models = []  # List of Model objects
        self.models_info = []  # List of Model RunInfos - respective to the models in self.models

        self.fusion_method = None  # Model object
        self.fusion_info = None  # Fusion RunInfo for the self.fusion_method

        self.analysis = EnsembleAnalysis(perf_metric, eval_perf_metrics, eval_div_metrics,
                                         print_results=True)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test: pd.DataFrame = None,
            y_test: pd.DataFrame = None, train_size: float = 0.6, valid_size: float = 0.3,
            test_size: float = 0.1, stratify: list = None, n_processes=4):
        """
        Fits the ensemble

        :param X_train: The training data
        :param y_train: The training labels
        :param X_valid: The validation data
        :param y_valid: The validation labels
        :param X_test: The test data
        :param y_test: The test labels
        :param train_size: Fraction of the given data used as the training set.
        :param valid_size: Fraction of the given data used as the validation set.
        :param test_size: Fraction of the given data used as the test set.
        :param stratify: Used for stratified sampling if not None.
                         Only used when train/valid split is executed.
        :param n_processes: Number of processes used for parallelization
        """

        logger.info("...Setting up")
        start = time.time_ns()
        self._set_up(X_train, y_train, X_valid, y_valid, X_test, y_test, train_size, valid_size,
                     test_size, stratify)
        logger.info("...Done")
        self._fit(n_processes)
        end = (time.time_ns() - start) / 1e+9
        self.analysis.runtime["complete"] = end

        self.data.clean_up()

    def predict(self, X: Union[pd.DataFrame, np.array]) -> np.array:
        """
        Predicts labels for given data.
        Uses the previously fitted ensemble model.

        :param X: A data matrix of shape (n_samples, n_features), for which the labels should be
                  predicted.
        :return: A matrix of shape (n_samples, n_classes) representing the predictions.
        """
        # First predict Models

        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        predictions = [model.predict(X) for model in self.models]
        # If the problem is NOT multi-label or the combiner does NOT take probability inputs,
        # The predictions have to be transformed to tensor form, so that the combiner can use them.
        if MULTILABEL not in self.specification['problem']:
            predictions = transform_label_tensor_to_class_assignment_tensor(predictions,
                                                                            self.data.n_classes)

        # Then Combine
        fused_prediction = self.fusion_method.predict(predictions)
        label_prediction = multiclass_assignments_to_labels(fused_prediction)
        return self.data.enc.inverse_transform(label_prediction)

    def evaluate(self, X_test, y_test: np.array, metric: list, n_classes: int = None,
                 filepath: str = "", print_results: bool = True):
        """
        Given a test data and a ground truth, compute the values for the metrics.

        :param X_test: The test data
        :param y_test: The ground truth
        :param metric: The metrics to use for the evaluation
        :param n_classes: Number of classes in the data. If None it will be inferred from y_true
        :param filepath: Filepath where the evaluation should be stored. If "" it will not be saved.
        :param print_results: If True the evaluation will be printed.
        :return: A dictionary containing the metrics and their values
        """

        if n_classes is None:
            n_classes = np.unique(y_test).tolist()

        self.analysis.set_data([X_test, y_test, n_classes])

        self.analysis.filepath = filepath
        self.analysis.print_results = print_results
        self.analysis.performance_metrics = metric

        self.analysis.set_combiner(self.fusion_info, self.fusion_method)
        self.analysis.set_base_classifier(self.models_info, self.models)
        self.analysis.evaluate_performance()
        self.analysis.evaluate_runtime()

    def fit_evaluate(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                     X_valid: pd.DataFrame = None, y_valid: pd.DataFrame = None,
                     X_test: pd.DataFrame = None, y_test: pd.DataFrame = None,
                     train_size: float = 0.6, valid_size: float = 0.3, test_size: float = 0.1,
                     stratify: list = None, save_path: str = "", n_processes=4):
        """
        Fits the ensemble on the given data and prints the values of the performance metrics.

        :param X_train: The training data
        :param y_train: The training labels
        :param X_valid: The validation data
        :param y_valid: The validation labels
        :param X_test: The test data
        :param y_test: The test labels
        :param train_size: Fraction of the given data used as the training set.
        :param valid_size: Fraction of the given data used as the validation set.
        :param test_size: Fraction of the given data used as the test set.
        :param stratify: Used for stratified sampling if not None.
                         Only used when train/valid split is executed.
        :param save_path: If a string is passed, saves the evaluation results to this path.
        :param n_processes: Number of processes used for parallelization
        """
        self.analysis.set_filepath(save_path)

        # Get the data
        logger.info("...Setting up")
        start = time.time_ns()
        self.data = self._set_up(X_train, y_train, X_valid, y_valid, X_test, y_test, train_size,
                                 valid_size, test_size, stratify)
        logger.info("...Done")  #

        # Fit the model
        self._fit(n_processes)
        logger.info("...Evaluating Ensemble")
        end = (time.time_ns() - start) / 1e+9
        self.analysis.runtime["complete"] = end
        self._run_evaluation()

    def save_ensemble(self, filename: str):
        """
        Pickles the ensemble and saves it to files/models/filename

        :param filename: filename of the pickled ensemble
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file, -1)

    def _set_up(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame,
                y_valid: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
                train_size: float, valid_size: float, test_size: float,
                stratify: list) -> DataStore:
        """
        Sets the parameters and data for the ensemble learner.

        :param X_train: The training data used for fitting the model.
        :param y_train: The training labels for the data in X_train.
        :param X_valid: The validation data used for fitting the model.
        :param y_valid: The validation labels for the data in X_valid.
        :param X_test: The test data
        :param y_test: The test labels
        :param options: Options used to determine the given problem and the ML-
                        and fusion algorithms to use.
                        'auto': If set to 'auto', the problem and algorithms will be selected
                        automatically based on the given training data and the feature types.
                        Specification : If a Specification object is passed, then the values set in
                        that object will be used as the problem and algorithms.
                        It is assumed, that the given specification and algorithms fit together.
        :param train_size: Fraction of the given data used as the training set.
        :param valid_size: Fraction of the given data used as the validation set.
        :param test_size: Fraction of the given data used as the test set.
        :param stratify: Used for stratified sampling if not None.
                         Only used when train/valid split is executed.
        :return: X_train, y_train, X_valid, y_valid, X_test, y_test
        """
        # Get specification
        self.classifiers, self.specification = generate_specification(X_train, y_train)

        # Prepare data
        self.data = DataStore(X_train, y_train, X_valid, y_valid, X_test, y_test, train_size,
                              valid_size, test_size, stratify, self.n_splits, seed=self.seed)

        return self.data

    def _fit(self, n_processes: int = 1):
        """
        Fits the ensemble

        :param n_processes: Number of processes used for parallelization
        """

        # Compute classification models
        clf_models = self._optimize_classifiers(n_processes)

        # Prepare Selection
        start = time.time_ns()
        selector = ClusterSelection(self.div_metric, self.perf_metric, None,
                                    n_classes=self.data.n_classes, heuristic=self.sel_heuristic,
                                    n_processes=n_processes)
        selector.prepare_diversity(clf_models, self.data.y_valid)
        end = (time.time_ns() - start) / 1e+9
        logger.debug("Needed %f sec. to compute the diversity for all model pairs.", end)
        self.analysis.runtime["selection_prep"] = end

        #  Find best ensemble
        logger.info("...Computing Ensemble")
        models, fusion_method = self._compute_best_ensemble(selector, n_processes)
        # Retrain the ensemble
        logger.info("Retraining base models")
        start = time.time_ns()
        self.models, self.fusion_method = retrain_ensemble(self.data, models, fusion_method)
        self.models_info, self.fusion_info = models, fusion_method
        self.analysis.runtime["retraining"] += (time.time_ns() - start) / 1e+9
        logger.info("...Best Ensemble size : %i", len(self.models))
        logger.info("...Best Fusion Method : %s", self.fusion_method.algorithm)
        logger.info("...Done")

    def _optimize_classifiers(self, n_processes):
        """
        Compute the pool of possible classifiers for the ensemble, by running a optimization over
        a configuration space of classification algorithms.

        :return
        """
        logger.info("...Computing ML-Models")
        cs_models = create_cs(self.classifiers, has_cat=len(self.data.cat_mask) > 0, default_value="RF")
        evaluator_models = ClassificationEvaluator(self.data, [self.perf_metric],
                                                   self.perf_metric, self.n_splits)
        model_history = self._run_hpo(cs_models, self.budget_m, evaluator_models, n_processes)
        self.analysis.classifier_history = model_history
        n_models = len(model_history.get_history_list())

        logger.info("...Evaluated %i Classifier Configurations Successfully", n_models)
        if n_models < 2:
            logger.error("Less then two models were trained. Increase budget for model generation.")
            raise ValueError

        if isinstance(self.ensemble_size,
                      int) and self.sel_heuristic != SILHOUETTE and n_models < self.ensemble_size:
            logger.warning("Given Ensemble size is larger then models found. Setting ensemble size "
                           "as amount of models found.")
            self.ensemble_size = n_models
        logger.info("...Done")

        return model_history

    def _compute_best_ensemble(self, selector: Selector, n_processes: int) -> (
            list, (Type[BaseMethod], RunInfo)):
        """
        Find the best ensemble including ensemble size and optimal fusion method

        :param selector: The selection object holding the models and diversity values.
        :param n_processes: Number of processes to use.
        :return: List of Models of the best ensemble, Best Fusion Method and its RunInfo
        """
        best_models, best_fusion_info = None, None

        # Find the best ensemble size
        if isinstance(self.ensemble_size, int):
            min_size = 2 if self.find_ensemble_size and self.sel_heuristic != SILHOUETTE else self.ensemble_size
            size_range = range(min_size, self.ensemble_size + 1)
        else:
            size_range = [self.ensemble_size[1]]

        for k in size_range:
            logger.info("...Selecting %i models", k)
            # Select models
            fusion_info, fusion_history, models = self._compute_ensemble(selector, k, n_processes)
            if best_fusion_info is None or fusion_info.cost[self.perf_metric] < \
                    best_fusion_info.cost[self.perf_metric]:
                best_fusion_info = fusion_info

                if not isinstance(self.ensemble_size, Tuple):
                    best_models = models
                else:
                    best_models = selector.select(fusion_info.configuration['ENS:size'])
                self.analysis.cmb_history = fusion_history

            self.analysis.size_history[k] = (fusion_info, models)

        if best_fusion_info is None:
            logger.error("Not enough fusion methods trained. Increase budget.")
            raise ValueError

        return best_models, best_fusion_info

    def _compute_ensemble(self, selector: Selector, k: int, n_processes: int) -> (
            (Type[BaseMethod], RunInfo), list, list):
        """
        Selects the best combiner and models for a given ensemble size k.

        :param selector: The selection object holding the models and diversity values.
        :param k: Selected ensemble size - irrelevant, if it will be optimized with AutoML
        :param n_processes: Number of processes to use.
        :return: Combiner with minimum cost, its RunInfo, the complete RunHistory and the models
        """

        sel = None
        if isinstance(self.ensemble_size, Tuple):
            k = self.ensemble_size
            sel = selector

        self.combiner_simple, self.combiner_hp = get_combiner(self.specification)

        cs_combiner = create_cs(self.combiner_hp + self.combiner_simple, ens_size=k,
                                default_value="DTEMP")

        # Compute fusion method
        logger.info("...Computing Fusion-Methods")
        evaluator_combiner = CombinerEvaluator(self.data, [self.perf_metric], self.perf_metric,
                                               self.n_splits, sel)
        if sel is None:
            evaluator_combiner.set_classifiers(selector.select(k))
        combiner_history = self._run_hpo(cs_combiner, self.budget_f, evaluator_combiner,
                                         n_processes)
        logger.info("...Evaluated %i Fusion Configurations Successfully",
                    len(combiner_history.get_history_list()))

        # Select and return the best Fusion Method and the associated models
        selected_combiner = combiner_history.get_lowest_cost_run(self.perf_metric)
        models = selector.select(selected_combiner.configuration['ENS:size'])

        return selected_combiner, combiner_history, models

    def _run_hpo(self, c_space: ConfigurationSpace, budget: int, evaluator: BaseEvaluator,
                 n_processes: int = 4) -> History:
        """
        Runs the hyperparameter optimizer on a given configuration space and for a given evaluator.

        :param c_space: The Configuration Space the hyperparameter optimizer should search for
                   configurations.
        :param budget: The time budget. States for how many seconds the HPO should run.
        :param evaluator: The Evaluation class, which is used to evaluate a generated configuration.
        :param n_processes: The number of processes used to execute the HPO.
        :return: The history of configurations evaluated by the HPO.
        """

        solver = self.solver(c_space, budget, evaluator, seed=self.seed)
        start = time.time_ns()
        solver.run(self.cutoff_time, n_processes)
        end = (time.time_ns() - start) / 1e+9
        logger.info("Needed %f seconds to run BO", end)
        logger.info("No. models trained: %s", str(len(solver.get_history().get_history_list())))

        if isinstance(evaluator, ClassificationEvaluator):
            self.analysis.runtime["model optimization"] = end
        elif isinstance(evaluator, CombinerEvaluator):
            self.analysis.runtime["fusion optimization"] = end
        else:
            self.analysis.runtime["model optimization"] = end
            self.analysis.runtime["fusion optimization"] = end
        return solver.get_history()

    def _run_evaluation(self):
        """
        Loads the necessary information into the evaluation object and runs the evaluation.
        """

        self.analysis.set_data(self.data)
        self.analysis.set_combiner(self.fusion_info, self.fusion_method)
        self.analysis.set_base_classifier(self.models_info, self.models)
        self.analysis.evaluate_performance()
        self.analysis.evaluate_diversity()
        self.analysis.evaluate_runtime()
        self.analysis.evaluate_best_family()

        self.analysis.evaluate_performance_ensemble(self.combiner_hp + self.combiner_simple,
                                                    self.solver, self.budget_f, self.n_splits,
                                                    self.seed)
