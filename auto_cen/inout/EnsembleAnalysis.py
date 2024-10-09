import os
from itertools import combinations
from typing import List, Union

import numpy as np
import sklearn.exceptions

from auto_cen.inout.evaluation_utils import save_csv, create_table, model_as_row, \
    preprocess_metrics
from auto_cen.main.cs_creator import create_cs
from auto_cen.main.data_storage import DataStore
from auto_cen.main.utils import retrain_ensemble
from auto_cen.optimization.history import RunInfo
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.pipeline.evaluation.evaluator_fusion import CombinerEvaluator
from auto_cen.pipeline.generic_clf_pipeline import ClassifierPipeline
from auto_cen.pipeline.generic_cmb_pipeline import CombinerPipeline
from auto_cen.pusion.utils import transform_label_tensor_to_class_assignment_tensor, \
    transform_label_vector_to_class_assignment_matrix
from auto_cen.utils.diversity_metrics import compute_diversity_metrics, get_oracle_predictions
from auto_cen.utils.performance_metrics import get_performance_report
from auto_cen.utils.utils import process_configuration


class EnsembleAnalysis:
    """
    Class for analysing the created ensemble with the data used in the optimization procedure.
    """

    def __init__(self, opt_metric: str, performance_metrics: list, diversity_metrics: list,
                 filepath: str = "", print_results: bool = False):

        self.base_classifier: List[RunInfo] = []
        self.base_classifier_models: List[ClassifierPipeline] = []
        self.combiner: RunInfo = None
        self.combiner_model: BaseMethod = None
        self.data = None

        self.classifier_history = None
        self.cmb_history = None
        self.size_history = {}

        self.opt_metric = opt_metric
        self.performance_metrics: list = performance_metrics
        self.diversity_metrics: list = diversity_metrics

        self.best_cmb_per_family = {}

        self.runtime: dict = {'model optimization': 0, 'selection_prep': 0,
                              'fusion optimization': 0, 'retraining': 0, 'complete': 0}

        self.filepath = filepath
        self.print_results = print_results

    def evaluate_performance(self):
        """
        Evaluate the created ensemble.

        """

        cmb_metrics, bc_metrics = predict_evaluate(self.data, self.combiner_model,
                                                   self.base_classifier_models,
                                                   self.performance_metrics, True)

        if self.print_results:
            output_table = self._create_evaluation_table(cmb_metrics, bc_metrics)
            print(output_table)

        if self.filepath != "":
            k_intended = self.combiner.configuration['ENS:size']
            self._save_evaluation(k_intended, cmb_metrics, bc_metrics, "ensembles.csv")

    def _create_evaluation_table(self, cmb_metrics, bc_metrics,
                                 title: str = "Evaluation Results") -> str:
        """
        Create a table as a string, containing the performance values of the ensembles.

        :param cmb_metrics: The performance metrics of the fusion method.
        :param bc_metrics The performance metrics of all models.
        :return: The table string
        """

        p = ['Enc', 'Pre', 'Fe', 'Div']
        clf_rows = []
        # Create the rows for the classifiers
        for i, c in enumerate(self.base_classifier):
            clf_config = process_configuration([c.algorithm] + p, c.algorithm, c.configuration)
            clf_config = [v[0] if k != "Enc" or len(v[1]) < 1 else v[1]["encoder"] for k, v in
                          clf_config.items()]
            clf_metrics = preprocess_metrics(bc_metrics[i])
            clf_rows.append(clf_config + clf_metrics)

        # Create the row for the combiner
        cmb_row = [[self.combiner.algorithm, "", "", "", "", *preprocess_metrics(cmb_metrics)]]

        header = ["Algorithm", "Encoder", "PreProc", "FE", "Diversity", *cmb_metrics.keys()]
        sections = [["Combiner", cmb_row], ["Classifiers", clf_rows]]

        return create_table(title, header, sections)

    def _save_evaluation(self, k_intended, cmb_metrics, bc_metrics, filename="ensembles.csv"):
        """
        For all evaluated ensembles write here the used models.

        :param k_intended: The wanted ensemble size.
        :param cmb_metrics: The performance metrics of the fusion method.
        :param bc_metrics The performance metrics of all models.
        """

        data = []
        header = ["model type", "k", "wanted k", "rid", "algorithm", "configuration",
                  "runtime"] + [*cmb_metrics.keys()]

        k = len(self.base_classifier)
        data.append(model_as_row(self.combiner, cmb_metrics, ["Fusion", k, k_intended]))
        for i, clf in enumerate(self.base_classifier):
            data.append(model_as_row(clf, bc_metrics[i], ["Model", k, k_intended]))

        save_csv(data, header, self.filepath + "/" + filename)

    def _get_best_family(self) -> tuple:
        """
        Returns the best model for each fusion algorithm.
        :return:  A list of RunInfo objects and their respective evaluation results
        """
        best_family = []
        perf_results = []
        history = self.cmb_history.get_history()
        X, y, _, _ = self.data.compute_fusion_data(self.base_classifier)

        for k in history.keys():
            models = history[k]
            best = min(models, key=lambda x: x.cost[self.opt_metric])
            try:
                model = CombinerPipeline(best.algorithm, best.configuration, seed=best.rnd_seed)
                model.fit(X, y)
                cmb_metrics, _ = predict_evaluate(self.data, model, self.base_classifier_models,
                                                  self.performance_metrics, get_clf_metrics=False)
            except sklearn.exceptions.ConvergenceWarning:
                print("%s could not converge. Set metrics as -1.", best.algorithm)
                cmb_metrics = {metric: -1 for metric in self.performance_metrics}
            perf_results.append(cmb_metrics)
            best_family.append(best)

        return best_family, perf_results

    def evaluate_best_family(self):
        """
        Evaluate the best model for each decision fusion algorithm.
        """
        family, fam_metrics = self._get_best_family()
        iterate = range(len(family))

        if self.print_results:
            columns = ["Algorithm", *self.performance_metrics]
            rows = [[family[i].algorithm, *preprocess_metrics(fam_metrics[i])] for i in iterate]
            output_table = create_table("Decision Fusion Family", columns, [["", rows]])
            print(output_table)

        if self.filepath != "":
            header = ["rid", "algorithm", "configuration", "runtime"] + self.performance_metrics
            rows = [model_as_row(family[i], fam_metrics[i]) for i in iterate]
            save_csv(rows, header, self.filepath + "/best_family.csv")

    def _compute_ensemble_diversity(self, models, model_infos, y_true) -> (float, List):
        """
        Recompute the diversity of the models.

        :param y_true: y_test
        :return: A tuple containing the diversity of the ensemble and a list for the diversity pairs
        """

        diversity_pairs = []
        ensemble_diversity = np.zeros(len(self.diversity_metrics))
        clf_pred = [clf.predict(self.data.X_test) for clf in models]
        diff = [get_oracle_predictions(pred, y_true) for pred in clf_pred]
        pairs_idx = list(combinations(range(0, len(models)), 2))

        for pair in pairs_idx:
            idx1, idx2 = pair
            divs = compute_diversity_metrics(diff[idx1], diff[idx2], clf_pred[idx1], clf_pred[idx2],
                                             y_true, self.diversity_metrics)
            ensemble_diversity += list(divs.values())
            diversity_pairs.append([model_infos[idx1], model_infos[idx2], divs])

        ensemble_diversity = ensemble_diversity / len(pairs_idx)

        return np.around(ensemble_diversity, 3), diversity_pairs

    def evaluate_diversity(self):
        """
        Print the diversity of the ensemble and its classifiers
        """
        ens_div, div_pairs = self._compute_ensemble_diversity(self.base_classifier_models,
                                                              self.base_classifier,
                                                              self.data.y_test)

        if self.print_results:
            header = ["Algorithm Pair"] + self.diversity_metrics
            ensemble_row = ["", [[self.combiner.algorithm, *ens_div]]]
            clf_rows = ["", [(pair[0].algorithm + " + " + pair[1].algorithm,
                              *np.around(list(pair[2].values()), 3)) for pair in div_pairs]]
            print(create_table("Ensemble Diversity", header, [ensemble_row, clf_rows]))

        if self.filepath != "":
            header = ["m1 rid", "m1 algorithm", "m1 configuration", "m2 rid", "m2 algorithm",
                      "m2 configuration", "m2 runtime"] + self.diversity_metrics
            rows = [model_as_row(pair[1], pair[2], [pair[0].rid, pair[0].algorithm,
                                                    pair[0].configuration]) for pair in div_pairs]
            save_csv(rows, header, self.filepath + "/diversity.csv")

    def evaluate_runtime(self):
        """
        Evaluate the runtime of the framework.
        """

        if self.print_results:
            output_table = create_table("Runtime", ["Framework Step", "Runtime in (sec)"],
                                        [["Steps", [(k, v) for k, v in self.runtime.items()]]])
            print(output_table)

        if self.filepath != "":
            save_csv([list(self.runtime.values())], list(self.runtime.keys()),
                     self.filepath + "/runtime.csv")

    def evaluate_ensemble_size(self):
        """
        Evaluate the different ensemble sizes that were tried.
        """
        size_metrics = []

        for k, v in self.size_history.items():
            try:
                k_models, k_cmb = retrain_ensemble(self.data, v[1], v[0])
                cmb_metrics, _ = predict_evaluate(self.data, k_cmb, k_models,
                                                  self.performance_metrics, get_clf_metrics=False)
                ens_div, _ = self._compute_ensemble_diversity(k_models, v[1], self.data.y_test)
            except:
                print("Something went wrong. Set Metrics as -1")
                cmb_metrics = {metric: -1 for metric in self.performance_metrics}
                ens_div = np.full(len(self.diversity_metrics), -1)
            size_metrics.append([k, v[0], cmb_metrics, list(ens_div), len(v[1])])

        if self.print_results:
            rows = [[row[0], row[-1], row[1].algorithm, *preprocess_metrics(row[2]), *row[3]] for
                    row
                    in
                    size_metrics]
            output_table = create_table("Ensemble Size",
                                        ["Ensemble Size", "True Size", "algorithm",
                                         *self.performance_metrics, *self.diversity_metrics],
                                        [["", rows]])
            print(output_table)
        if self.filepath != "":
            header = ["size", "true size", "rid", "algorithm", "configuration",
                      "runtime"] + self.performance_metrics + self.diversity_metrics
            rows = [model_as_row(row[1], row[2], [row[0], row[-1]]) + row[3] for row in
                    size_metrics]
            save_csv(rows, header, self.filepath + "/ensemble_sizes.csv")

    def evaluate_performance_ensemble(self, combiner_space, solver, budget, n_splits, seed):
        """
        Compute the best ensemble of the same size without diversity optimization.
        """

        # Compute the performance ensemble
        perf_infos = self.classifier_history.get_cost_sorted_history(self.opt_metric)[
                     :len(self.base_classifier)]
        perf_config = self._reoptimize_ensemble(budget, self.data, perf_infos, combiner_space,
                                                solver, n_splits, seed)
        X, y, _, _ = self.data.compute_fusion_data(perf_infos)
        perf_models, perf_cmb = retrain_ensemble(self.data, perf_infos, perf_config)
        perf_cmb_metrics, perf_model_metrics = predict_evaluate(self.data, perf_cmb, perf_models,
                                                                self.performance_metrics,
                                                                get_clf_metrics=True)

        ens_div, _ = self._compute_ensemble_diversity(perf_models, perf_infos, self.data.y_test)

        if self.print_results:
            output_table = self._create_evaluation_table(perf_cmb_metrics, perf_model_metrics,
                                                         "Performance Results")
            print(output_table)

            header = ["Algorithm"] + self.diversity_metrics
            ensemble_row = ["", [[perf_config.algorithm, *ens_div]]]
            print(create_table("Performance Ensemble Diversity", header, [ensemble_row]))

        if self.filepath != "":
            k_intended = self.combiner.configuration['ENS:size']
            self._save_evaluation(k_intended, perf_cmb_metrics, perf_model_metrics,
                                  "performance.csv")

            header = ["Algorithm"] + self.diversity_metrics
            rows = [[perf_config.algorithm, *ens_div]]
            save_csv(rows, header, self.filepath + "/performance_diversity.csv")

    def _create_comparison_table(self, static_fusion: list, sf_results: list, perf_ens: RunInfo,
                                 perf_metrics: dict, best_model: RunInfo,
                                 best_clf_metrics: dict, cmb_metrics: dict) -> str:
        """
        Create a table as a string, containing the values for the ensemble, performance,
         static fusion and best classifier.

        :param static_fusion: List of fusion methods that should be compared against.
        :param perf_ens: An ensemble but without diversity optimisation.
        :param best_model: RunInfo object of the best classifier.
        :param best_clf_metrics: The performance metrics of the best classifier.
        :return: The table string.
        """

        header = ["Type", "Algorithm", *self.performance_metrics]
        rows = [["Optimized", self.combiner.algorithm, *preprocess_metrics(cmb_metrics)],
                ["Best Model", best_model.algorithm, *preprocess_metrics(best_clf_metrics)],
                ["Performance", perf_ens.algorithm, *preprocess_metrics(perf_metrics)]]
        for i, sf in enumerate(static_fusion):
            rows.append(["Static Fusion", sf.algorithm, *preprocess_metrics(sf_results[i])])

        return create_table("Comparison", header, [["", rows]])

    def _save_comparison_evaluation(self, static_fusion: list, sf_results: list, perf_ens: RunInfo,
                                    perf_models: List[RunInfo], perf_metric: dict,
                                    best_model: RunInfo, best_clf_metrics: dict, cmb_metrics: dict):
        """
        Save the comparison to a pickle file.

        :param static_fusion: List of fusion methods that should be compared against.
        :param perf_ens: An ensemble but without diversity optimisation.
        :param perf_models: The classifiers used by the performance ensemble.
        :param best_model: RunInfo object of the best classifier.
        :param best_clf_metrics: The performance metrics of the best classifier.
        """

        header = ["evaluation type", "model ids", "rid", "algorithm", "configuration",
                  "runtime"] + self.performance_metrics

        standard_models = [clf.rid for clf in self.base_classifier]
        perf_rids = [pclf.rid for pclf in perf_models]
        rows = [model_as_row(self.combiner, cmb_metrics, ["standard", standard_models]),
                model_as_row(best_model, best_clf_metrics, ["best model", []]),
                model_as_row(perf_ens, perf_metric, ["performance", perf_rids])]

        for i, sf in enumerate(static_fusion):
            rows.append(model_as_row(sf, sf_results[i], ["static fusion", standard_models]))

        save_csv(rows, header, self.filepath + "/comparison.csv")

    def _reoptimize_ensemble(self, budget: int, data: DataStore, models: list, config_space: list,
                             solver, n_splits: int, seed) -> RunInfo:
        """
        Recompute the decision fusion of an ensemble.

        :param budget: The optimization budget in seconds.
        :param data: The data used for training and evaluation.
        :param models: List of the classifiers used.
        :param config_space: List of algorithms used for the creation of the configuration space.
        :param solver: Solver object used for optimization.
        :param n_splits: Number of splits used for CV.
        :return: The decision fusion model.
        """
        cs = create_cs(config_space, len(self.base_classifier))
        evaluator = CombinerEvaluator(data, [self.opt_metric], self.opt_metric, n_splits)
        evaluator.set_classifiers(models)
        slv = solver(cs, budget, evaluator, [], seed)
        slv.run()
        fusion = slv.get_history().get_lowest_cost_run(self.opt_metric)
        return fusion

    def set_base_classifier(self, base_classifiers_info: List[RunInfo],
                            base_classifiers_models: List[BaseMethod]):
        self.base_classifier = base_classifiers_info
        self.base_classifier_models = base_classifiers_models

    def set_combiner(self, combiner_info: List[RunInfo], combiner_model: List[BaseMethod]):
        self.combiner = combiner_info
        self.combiner_model = combiner_model

    def set_filepath(self, filepath):
        EVAL_PATH = "results/"
        if not os.path.exists(EVAL_PATH + filepath + "/"):
            os.makedirs(EVAL_PATH + filepath + "/")

        self.filepath = EVAL_PATH + filepath

    def set_data(self, data: Union[List, DataStore]):
        self.data = data


def predict_evaluate(data: Union[DataStore, List], fusion_method: BaseAlgorithm,
                     models: List[BaseAlgorithm], perf_metrics: list,
                     get_clf_metrics: bool = True) -> tuple:
    """
    Evaluates the complete ensemble including its models.

    :param data: Data which was used to train, evaluate and test the ensemble
    :param fusion_method: Object of the fusion method
    :param models: List of the models used by the ensemble
    :param perf_metrics: List of performance metrics computed for the evaluation
    :param get_clf_metrics: True, if the results of the individual classifiers should also be returned.

    :return: Dict of metrics of the ensemble, list of dicts of the metrics for all models
    """
    model_eval = []

    X_test = data[0] if isinstance(data, List) else data.X_test
    y_test = data[1] if isinstance(data, List) else data.y_test
    n_classes = data[2] if isinstance(data, List) else data.n_classes

    # If not cross-validation, recompute the metrics
    predictions = []
    for model in models:
        predictions.append(model.predict(X_test))
    predictions = np.array(predictions)
    if get_clf_metrics:
        for i, pred in enumerate(predictions):
            metr = get_performance_report(y_test, pred, None, perf_metrics, n_classes)
            model_eval.append(metr)
    predictions = _get_predictions_for_fusion(predictions, n_classes, False)
    fused_prediction = fusion_method.predict(predictions)

    # Then compute evaluation metrics for the whole ensemble
    ens_metrics = _compute_ens_metrics(y_test, fused_prediction, perf_metrics, n_classes, False)

    return ens_metrics, model_eval


def cost_to_performance(cost_old: dict) -> dict:
    """
    Converts the cost (i.e. a loss) into the original performance metric.
    This is simply done by computing 'performance=1-cost'.

    :param cost_old: A dictionary holding the old cost.

    :return: The performance metrics as a dictionary.
    """
    perf = cost_old.copy()
    for k, v in perf.items():
        perf[k] = 1 - v
    return perf


def _get_predictions_for_fusion(predictions: np.array, n_classes: int,
                                is_multilabel: bool) -> np.array:
    """
    Checks if the predictions have to be transformed for using them with fusion methods.
    If yes they are transformed.

    :param predictions: The model predictions.
    :param n_classes: The number of classes in the problem.
    :param is_multilabel: True, if it is a multilabel problem.
    :return: The (transformed) predictions
    """
    if not is_multilabel and len(predictions.shape) < 3:
        # If not multilabel and not prob predictions
        predictions = transform_label_tensor_to_class_assignment_tensor(predictions, n_classes)
    if is_multilabel and len(predictions.shape) > 2:
        # If not multilabel and not prob predictions
        predictions = [[np.argmax(smpl, axis=1) for smpl in clf] for clf in predictions]

    return predictions


def _compute_ens_metrics(y_test: np.array, fused_prediction: np.array, perf_metrics: list,
                         n_classes: int, is_multilabel: bool) -> dict:
    """
    Compute the performance metrics for the ensemble.

    :param y_test: The test data.
    :param fused_prediction: The predictions of the ensemble.
    :param perf_metrics: The performance metrics
    :param n_classes: The number of classes.
    :param is_multilabel: True, if it is a multilabel problem.
    :return: The computed metrics as a dict, where each metric is the key and its value is its
             computed value.
    """
    y_test_transformed = y_test
    if not is_multilabel:
        y_test_transformed = transform_label_vector_to_class_assignment_matrix(y_test,
                                                                               n_classes)
    ens_metrics = get_performance_report(y_test_transformed, fused_prediction, fused_prediction,
                                         perf_metrics, n_classes)

    return ens_metrics
