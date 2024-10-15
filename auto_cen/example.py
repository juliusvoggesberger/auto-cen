"""
Runs the AutoML Framework
"""

import sys
import logging

from openml import datasets

from auto_cen.constants import ACCURACY, F1_MACRO, BALANCED_ACCURACY, DOUBLEFAULT, ROC_AUC_OVO, \
    DISAGREEMENT, DOUBLEFAULT_NORM
import auto_cen as ao
from auto_cen.optimization.bo import BayesianOptimization

# Activate Logging
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('auto_cen')
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    # Get a dataset from opemnl, for example kropt (184)
    dataset_id = 184
    dataset = datasets.get_dataset(dataset_id, download_data=True, download_qualities=False,
                                   download_features_meta_data=False)
    X, y, _, _ = dataset.get_data(dataset.default_target_attribute)

    # Set the input parameters
    dataset = str(dataset_id)
    solver = "BO"
    budget_m = 60
    budget_f = 60
    n_splits = 10
    cutoff_time = 60
    sel_heuristic = ""
    perf_metric = BALANCED_ACCURACY
    div_metric = DOUBLEFAULT_NORM
    ensemble_size = 10
    find_ensemble_size = False
    seed = 123

    # Set the foldername for the outputs
    SAVE_PATH = dataset + "_" + solver + "_M" + str(budget_m) + "_CV" + str(n_splits) + "_F" + str(
        budget_f) + "_CO" + str(cutoff_time) + "_SH" + str(
        sel_heuristic) + "_PERF" + perf_metric + "_DIV" + div_metric + "_SIZE" + str(
        ensemble_size) + "_SEED" + str(seed)

    # Set up Auto-CEn
    el = ao.EnsembleLearner(ensemble_size, budget_m, budget_f,
                            cutoff_time=cutoff_time,
                            solver=BayesianOptimization,
                            n_splits=n_splits,
                            find_ensemble_size=find_ensemble_size,
                            perf_metric=perf_metric,
                            sel_heuristic=sel_heuristic,
                            div_metric=div_metric,
                            eval_perf_metrics=[ACCURACY, BALANCED_ACCURACY, F1_MACRO, ROC_AUC_OVO],
                            eval_div_metrics=[DOUBLEFAULT_NORM, DOUBLEFAULT, DISAGREEMENT],
                            seed=seed)

    # Fit and evaluate the ensemble model
    el.fit_evaluate(X, y, n_processes=4, save_path=SAVE_PATH, stratify=y,
                    train_size=0.8, valid_size=0.0, test_size=0.2)
