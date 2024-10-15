"""
Runs the AutoML Framework
"""

import sys
import logging

from openml import datasets

import auto_cen as ac
from auto_cen.constants import BALANCED_ACCURACY, DOUBLEFAULT, DOUBLEFAULT_NORM
from auto_cen.optimization.bo import BayesianOptimization

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('auto_cen')
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    dataset_id = int(sys.argv[1])
    model_budget = int(sys.argv[2])
    fusion_budget = int(sys.argv[3])
    cutoff_time = int(sys.argv[4])
    cross_valid = int(sys.argv[5])
    sel_heuristic = str(sys.argv[6])
    p_metric = str(sys.argv[7])
    d_metric = str(sys.argv[8])
    seed = int(sys.argv[9])
    ens_size = int(sys.argv[10])

    if sel_heuristic == "None":
        sel_heuristic = ""

    SAVE_PATH = str(dataset_id) + "_M" + str(model_budget) + "_F" + str(
        fusion_budget) + "_CV" + str(cross_valid) + "_CO" + str(
        cutoff_time) + "_IDIV" + "_SH" + str(sel_heuristic) + "_PERF" + p_metric + "_SIZE" + str(
        ens_size) + "_SEED" + str(seed)

    # get the openml data
    dataset = datasets.get_dataset(dataset_id, download_data=True, download_qualities=False,
                                   download_features_meta_data=False)
    X, y, _, _ = dataset.get_data(dataset.default_target_attribute)

    el = ac.EnsembleLearner(ens_size, model_budget, fusion_budget,
                            cutoff_time=cutoff_time,
                            solver=BayesianOptimization,
                            n_splits=cross_valid,
                            find_ensemble_size=True,
                            perf_metric=p_metric,
                            sel_heuristic=sel_heuristic,
                            div_metric=d_metric,
                            eval_perf_metrics=[BALANCED_ACCURACY],
                            eval_div_metrics=[DOUBLEFAULT_NORM, DOUBLEFAULT],
                            seed=seed)
    el.fit_evaluate(X, y, n_processes=8, save_path=SAVE_PATH, stratify=y,
                    train_size=0.8, valid_size=0.0, test_size=0.2)
