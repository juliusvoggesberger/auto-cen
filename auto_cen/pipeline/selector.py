"""
Contains implementation for classes that allow the selection of a subset of models.
"""
import math

from abc import ABC
from multiprocessing import Manager
from multiprocessing.pool import Pool
from functools import partial
from itertools import combinations
from operator import itemgetter
from typing import List

import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

from auto_cen.constants import DOUBLEFAULT, ACCURACY, SILHOUETTE
from auto_cen.optimization.history import History, RunInfo
from auto_cen.utils.diversity_metrics import compute_diversity_metrics, get_oracle_predictions, \
    kappa_error


class Selector(ABC):
    """
    Implements a generic abstract class for computing the diversity of models and select a subset.

    :param d_method: Diversity metric used to compute the diversity.
    :param p_metric: Performance metric used to compute the performance.
    """

    def __init__(self, d_method, p_metric):
        self.d_method = d_method
        self.p_metric = p_metric

    def prepare_diversity(self, history: History, y_valid: np.array):
        """
        Method for computing the diversity of given models.

        :param y_valid: Ground Truth of the model predictions.
        :param history: A History object of clf models from which the subset will be selected.
        """
        raise NotImplementedError

    def select(self, n_models: int) -> list:
        """
        Method for selecting a number of models.

        :param n_models: Number of models which should be selected.
                           Can be less than the given number, if not enough models can be found.
                           Equals the number of classifiers which will be returned.
        :return: A list of models.
        """
        raise NotImplementedError


class ClusterSelection(Selector):
    """
    Select a subset of models with the aid of diversity and performance metrics.
    The selection method used, is described in
    "An Approach to the Automatic Design of Multiple Classifier Systems",
    Giacinto, Giorgio and Roli, Fabio,2001

    The method works as follows:
        1. For each pair of classifiers a diversity metric (default: double-fault) is computed.
        2. Using the diversity metric as a distance measure, agglomerative clustering is performed
            to find clusters.(default: agglomerative clustering with average linkage is used).
        3. From each cluster the classifier which maximizes a performance metric (default: accuracy)
            is selected. -> Use here minimum metric, as a loss (error rate =1-accuracy) is used.
        4. Return the list of selected classifiers.

    :param d_method: Diversity metric which is used to compute the distances. Default: Doublefault.
    :param p_metric: Performance metric which is used to select the 'best' method per cluster.
    :param diversity_metrics: List of further diversity metrics that are computed.
    :param heuristic: If "Diversity": uses non-pairwise diversity to select a set of the x% models
                      (x=10/20/../100).
                      If "Performance": selects the best performing 50% of models.
                      If "Averaged": uses the average method from caruana et al.
                      If "OrientationOrdering": uses orientation ordering.
                      If None: Uses all models, except the ones that are worse than random.
    :param n_classes: The number of classes in the data.
    :param n_processes: The number of processes.
    """

    def __init__(self, d_method: str = DOUBLEFAULT, p_metric: str = ACCURACY,
                 diversity_metrics: list = None, heuristic=None, n_classes=-1, n_processes=1):
        super().__init__(d_method, p_metric)
        self.n_processes = n_processes
        self.n_classes = n_classes
        self.heuristic = heuristic

        self.diversity_metrics = diversity_metrics
        if self.diversity_metrics is None:
            self.diversity_metrics = [self.d_method]
        if self.d_method not in self.diversity_metrics:
            self.diversity_metrics.append(self.d_method)

        self.models = []
        self.linkage_matrix = np.empty(0)
        self.diversity_list = None

    def prepare_diversity(self, history: History, y_valid: np.array,
                          clustering_method: str = "average"):
        """
        Given a list of models (RunInfo objects), create all pairings and compute the diversity for
        each pairing. This is only done for the top 50% of the models to reduce excessive
        computation time and reduce the impact of the worst models

        :param history: A History object of clf models from which the subset will be selected.
        :param y_valid: Ground Truth of the model predictions.
        :param clustering_method: The method used for computing the distance in agglomerative
                                  clustering. Is either "single", "average", "complete", "weighted",
                                  "centroid", "median" or "ward".
                                  See the scipy documentation for more details.
        """

        self.models, differences = self._heuristic(history, y_valid)
        predictions = [np.asarray(model.prediction_va) for model in self.models]
        # Instead of computing the pairings of all models, compute the pairings of their indices.
        pairs_idx = list(combinations(range(0, len(self.models)), 2))

        if self.n_processes > 1:
            # As making the diversity metric computation parallel would create a massive overhead
            # and nullify nearly all performance advantage, split the pairs instead into
            # n=n_processes chunks and make that parallel
            no_per_process = math.ceil(len(pairs_idx) / self.n_processes)
            chunks_idx = [pairs_idx[i:i + no_per_process] for i in
                          range(0, len(pairs_idx), no_per_process)]

            manager = Manager()
            shared_diff = manager.list(differences)

            with Pool(self.n_processes) as pool:
                # Already comes out sorted, so no need to take care of that
                chunked_d_list = pool.map(
                    partial(_chunked_diversity_list, diff=shared_diff, pred=predictions,
                            y_true=y_valid, metrics=self.diversity_metrics), chunks_idx)
            manager.shutdown()
            # Flatten list
            diversity_list = [d_pair for chunk in chunked_d_list for d_pair in chunk]
        else:
            diversity_list = _chunked_diversity_list(pairs_idx, differences, predictions, y_valid,
                                                     self.diversity_metrics)
        self.diversity_list = np.asarray([pair[self.d_method] for pair in diversity_list])
        # Scipy wants the upper triangle of the distance matrix condensed into a 1D list
        self.linkage_matrix = linkage(self.diversity_list, method=clustering_method)

    def select(self, n_models: int) -> List[RunInfo]:
        """
        Selects a subset of models, using their diversity and performance.


        :param n_models: Number of clusters which should be found.
                           Can be less than the given number, if not enough clusters can be found.
                           Equals the number of classifiers which will be returned.

        :return: A list containing the selected models.
        """

        # If the searched number of models is larger or the same as the found models, just return
        # them all.
        if n_models >= len(self.models) and self.heuristic != SILHOUETTE:
            return self.models
        if self.heuristic == SILHOUETTE:
            n_models = self._select_silhouette(n_models)
        clu_labels = fcluster(self.linkage_matrix, t=n_models, criterion='maxclust')

        selection = []

        for i in range(1, max(clu_labels) + 1):
            # Take the pairs of the i-th cluster
            # np.where returns a tuple, just take the first entry
            clu_ix = np.where(clu_labels == i)[0]
            if len(clu_ix) > 1:
                # If more than one model is in the cluster, select the model with minimum cost
                clu_models = list(itemgetter(*clu_ix)(self.models))
                best_model = clu_models[
                    np.argmin([model.cost[self.p_metric] for model in clu_models])]
            else:
                # If only one model is in the cluster
                best_model = self.models[clu_ix[0]]
            selection.append(best_model)
        return selection

    def _heuristic(self, history: History, y_valid: np.array) -> (List[RunInfo], np.array):
        """
        A heuristic that selects a subset of models.
        If no heuristic is selected, the models that are worse than random are removed.
        Else a heuristic is used, which is either a diversity-based heuristic,
        two performance-based heuristics (Top x%, top Avg.%) or orientation ordering.

        :param history: The Run History
        :param y_valid: The validation labels
        :return: The models and the oracle/difference outputs for each model
                 (Multilabel -> True/False)
        """
        models = history.get_cost_sorted_history(self.p_metric)
        differences = np.array(
            [get_oracle_predictions(model.prediction_va, y_valid) for model in models])

        if self.heuristic == "Diversity":
            models = _div_heuristic(differences, models)
        elif self.heuristic == "Averaged":
            models = _avg_heuristic(differences, models)
        elif self.heuristic == "OrientationOrdering":
            models = _orientation_ordering(differences, models)
        elif self.heuristic == "Performance":
            # Only use top 50% of models for selection if enough models were evaluated
            models = models[:math.ceil(len(models) / 2)]
        else:
            # Remove all models that are worse than random from the list
            # Stop if a model is found that is better than random
            for model in reversed(models):
                if (1 - model.cost[self.p_metric]) >= (1.0 / self.n_classes):
                    break
                models.remove(model)
        best_diff = [get_oracle_predictions(model.prediction_va, y_valid) for model in models]
        return models, best_diff

    def _select_silhouette(self, max_n_clusters):
        """
        Selects a subset of models, using their diversity and performance.
        Uses the silhouette measure to identify the "best" suited number of clusters.

        :param max_n_clusters:

        :return: The number of clusters selected by the approach.
        """
        max_silhouette = 0
        selected_n_clusters = 2

        # Reconstruct the full distance matrix
        full_matrix = np.zeros((len(self.models), len(self.models)), float)
        triu_idx = np.triu_indices(len(self.models), 1)
        full_matrix[triu_idx] = self.diversity_list
        full_matrix.T[triu_idx] = self.diversity_list
        for n_clusters in range(2, max_n_clusters + 1):
            clu_labels = fcluster(self.linkage_matrix, t=n_clusters, criterion='maxclust')
            silhouette = silhouette_score(full_matrix, clu_labels, metric="precomputed")
            if silhouette > max_silhouette:
                max_silhouette = silhouette
                selected_n_clusters = n_clusters
        return selected_n_clusters


def _div_heuristic(differences: np.array, models: list) -> list:
    """
    A heuristic based on a non-pairwise diversity metric.
    For the Top10%, Top20% .. Top100% of models according to their performance, and then computes a
    non-pairwise diversity metric on for that set.
    The set with the highest diversity is then selected.

    :param differences: The outputs of the models for the validation data.
                        Transformed into a True/False vector
    :param models: List of the models from which to select
    :return: A list of the selected models
    """
    best_kappa = 2
    best_ms = None
    for i in np.arange(0.1, 1.1, 0.1):
        # Select percentage of top performing models
        diff = np.array(differences[:math.ceil(len(differences) * i)])
        # Use kappa error instead of kohaw wolpert, as kuncheva says that is a simple to interpret
        # and as such good metric
        kappa = kappa_error(diff)
        if kappa < best_kappa:
            best_kappa = kappa
            best_ms = models[:math.ceil(len(models) * i)]

    return best_ms


def _avg_heuristic(differences: np.array, models: list) -> list:
    """
    Selects a subset by averaging the oracle predictions of the models.
    Taken from Caruana et al. "Getting the most out of ensemble selection", 2006.

    :param differences: The outputs of the models for the validation data. Transformed into a True/False vector
    :param models: List of the models from which to select
    :return: A list of the selected models
    """
    best_perf = 0
    best_ms = None
    for i in np.arange(0.1, 1.1, 0.1):
        diff = np.array(differences[:math.ceil(len(differences) * i)])
        # Compute accuracy for the oracle values (True/False)
        y_comb = np.around(np.sum(diff * 1.0, axis=0) / len(diff))
        perf = np.sum(y_comb) / len(y_comb)
        if perf >= best_perf:
            best_perf = perf
            best_ms = models[:math.ceil(len(models) * i)]
    return best_ms


def _orientation_ordering(differences: np.array, models: list) -> list:
    """
    Implements orientation ordering as suggested by
    "MARTÍNEZ-MUÑOZ, Gonzalo; SUÁREZ, Alberto. Pruning in ordered bagging ensembles. 2006"

    :param differences: The outputs of the models for the validation data.
                        Transformed into a True/False vector
    :param models: List of the models from which to select
    :return: A list of the selected models
    """
    csig = (differences * 2.0) - 1
    # o + lambda * cens
    cens = np.sum(csig, axis=0) / len(differences)
    o_vec = np.ones(len(cens))
    cref = o_vec + np.dot((-1. * o_vec), cens / np.dot(cens, cens)) * cens
    normed_cref = cref / np.linalg.norm(cref)
    normed_csig = csig / np.linalg.norm(csig, axis=1)[:, None]
    # Select the correct angles
    select = np.arccos(np.clip(np.dot(normed_csig, normed_cref), -1.0, 1.0)) < (np.pi / 2)
    indices = np.arange(len(select))[select]
    best_ms = [itemgetter(*indices)(models)] if len(indices) < 2 else list(
        itemgetter(*indices)(models))

    return best_ms


def _chunked_diversity_list(pair_chunk: list, diff: list, pred: list, y_true: np.array,
                            metrics: list) -> list:
    """
    Given a list of model pairs, compute the diversity measures for each pair.

    :param pair_chunk: A Pair of model indices
    :param diff: List of differences between the classifier predictions and the ground truth
    :param pred: List of the classifier predictions
    :param y_true: The ground truth
    :param metrics: List of diversity metrics
    :return: A list of tuples, where each tuple consists of (diversity, pair indices).
    """
    d_pair_list = [
        compute_diversity_metrics(diff[pair[0]], diff[pair[1]], pred[pair[0]], pred[pair[1]],
                                  y_true, metrics) for pair in pair_chunk]
    return d_pair_list
