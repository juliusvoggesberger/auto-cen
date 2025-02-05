import sys
import os
import time
from itertools import combinations

sys.path.insert(1, os.path.join(sys.path[0],
                                ".."))  # have to do this because of singularity. I hate it

from assembled_ensembles.utils.constants import DOUBLEFAULT_NORM, DOUBLEFAULT, DISAGREEMENT, \
    YULES_Q, CORRELATION_COEFFICIENT, YULES_Q_NORM, CORRELATION_COEFFICIENT_NORM

from pathlib import Path

from assembled.metatask import MetaTask
from assembled.ensemble_evaluation import evaluate_ensemble_on_metatask

from assembled_ensembles.util.config_mgmt import get_ensemble_switch_case_config
from assembled_ensembles.default_configurations.supported_metrics import msc

from assembled_ensembles.configspaces.evaluation_parameters_grid import get_config_space, \
    get_name_grid_mapping
from ConfigSpace import Configuration

from utils.diversity_metrics import get_oracle_predictions, compute_diversity_metrics
from utils.performance_metrics import create_performance_report

import pandas as pd
from sklearn import preprocessing


def create_table(title: str, columns: list, sections: list):
    blueprint = "{:<20}" * len(columns)
    header_divider = "=" * (18 * (len(columns) + 1)) + "\n"
    row_divider = "-" * (18 * (len(columns) + 1)) + "\n"

    # Title and header
    output_table = title + "\n" + header_divider
    output_table += blueprint.format(*columns) + "\n" + row_divider

    # Contents
    for section in sections:
        # Section Header
        if section[0] != "":
            output_table += section[0] + "\n"
        for row in section[1]:
            output_table += blueprint.format(*row) + "\n"
        output_table += row_divider
    return output_table


def save_csv(rows, header, filename):
    data_frame = pd.DataFrame(data=rows, columns=header)
    data_frame.to_csv(filename)


def compute_diversity(mt, filename, qdo):
    # TODO Currently uses ALL Predictors/Classifiers, not only the ones in the ensemble
    metrics = [DOUBLEFAULT_NORM, DOUBLEFAULT, DISAGREEMENT, YULES_Q, YULES_Q_NORM,
               CORRELATION_COEFFICIENT, CORRELATION_COEFFICIENT_NORM]
    ensemble_diversity = np.zeros(len(metrics))
    div_pairs = []
    # We only evaluate one fold anyways
    # Should be a pandas dataframe
    eval_data = mt.yield_evaluation_data([0])
    eval_data = [x for x in eval_data][0]
    y_train = eval_data[3]
    y_test = eval_data[4]
    clf_pred = eval_data[6]
    enc = preprocessing.LabelEncoder()
    y = pd.concat([y_train, y_test])
    enc.fit(y)
    y_true = enc.transform(y_test)
    clf_pred = enc.transform(clf_pred)
    models = mt.get_predictors_for_fold(0)
    clf_pred = np.transpose(clf_pred.to_numpy(np.int64))

    diff = [get_oracle_predictions(pred, y_true) for pred in clf_pred]
    pairs_idx = list(combinations(range(0, len(models)), 2))
    for _, _, X_test, _, _, _, test_base_predictions, _, test_base_confidences in mt.yield_evaluation_data(
            [0]):
        print(len(test_base_predictions))
        print(test_base_confidences)
        print("!!!")
    print(mt.predictor_descriptions)

    for pair in pairs_idx:
        idx1, idx2 = pair
        divs = compute_diversity_metrics(diff[idx1], diff[idx2], clf_pred[idx1], clf_pred[idx2],
                                         y_true, metrics)
        ensemble_diversity += list(divs.values())
        div_pairs.append([models[idx1], models[idx2], divs])

    # ensemble_diversity = np.around(ensemble_diversity / len(pairs_idx), 3)

    # header = ["Algorithm Pair"] + metrics
    # ensemble_row = ["", [["Ensemble", *ensemble_diversity]]]
    # clf_rows = ["", [(pair[0] + " + " + pair[1], *np.around(list(pair[2].values()), 3)) for pair in
    #                  div_pairs]]
    # print(create_table("Ensemble Diversity", header, [ensemble_row, clf_rows]))

    header = ["m1 algorithm", "m2 algorithm"] + metrics
    rows = [[pair[0], pair[1], *list(pair[2].values())] for pair in div_pairs]
    save_csv(rows, header, filename + "_diversity" + str(seed) + ".csv")


if __name__ == "__main__":
    # -- Get Input Parameter
    openml_task_id = sys.argv[1]
    pruner = sys.argv[2]  # "TopN", "SiloTopN"
    ensemble_method_name = sys.argv[3]
    metric_name = sys.argv[4]
    benchmark_name = sys.argv[5]
    evaluation_name = sys.argv[6]
    isolate_execution = sys.argv[7] == "yes"
    load_method = sys.argv[8]
    # folds_to_run_on = sys.argv[9]
    folds_to_run_on = [int(x) for x in sys.argv[9].split(",")] if "," in sys.argv[9] else [int(sys.argv[9])]
    config_space_name = sys.argv[10]
    ens_save_name = sys.argv[11]

    # New Parameters
    n_jobs = int(sys.argv[12])
    seed = int(sys.argv[13])

    if folds_to_run_on == "-1":
        folds_to_run_on = None
        state_ending = ""
    else:
        # folds_to_run_on = [int(folds_to_run_on)]
        state_ending = f"_{folds_to_run_on}"

    delayed_evaluation_load = True if load_method == "delayed" else False

    # -- Build Paths
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    tmp_input_dir = file_path.parent / "benchmark" / "input" / benchmark_name / pruner
    print("Path to Metatask: {}".format(tmp_input_dir))

    out_path = file_path.parent / "benchmark" / "output" / benchmark_name / "task_{}/{}/{}".format(
        openml_task_id,
        evaluation_name,
        pruner)
    out_path.mkdir(parents=True, exist_ok=True)

    s_path = file_path.parent / "benchmark/state/{}/task_{}/{}".format(benchmark_name,
                                                                       openml_task_id,
                                                                       evaluation_name)
    s_path.mkdir(parents=True, exist_ok=True)
    s_path = s_path / "{}_{}{}.done".format(pruner, ens_save_name, state_ending)
    print("Path to State: {}".format(s_path))

    # -- Rebuild The Metatask
    print("Load Metatask")
    # Enforce the file format to be feather (I had unknown and random issues with csv)
    mt = MetaTask(file_format="csv")
    mt.read_metatask_from_files(tmp_input_dir, openml_task_id,
                                delayed_evaluation_load=delayed_evaluation_load)
    # -- Setup Evaluation variables
    # Get the metric(s)
    is_binary = len(mt.class_labels) == 2
    # If the ensemble requires the metric, we assume the labels to be encoded
    ens_metric = msc(metric_name, is_binary, list(range(mt.n_classes)))
    # For the final score, we need the original labels
    score_metric = msc(metric_name, is_binary, mt.class_labels)
    predict_method = "predict_proba" if ens_metric.requires_confidences else "predict"

    # -- Handle Config Input
    cs = get_config_space(config_space_name)
    name_grid_mapping = get_name_grid_mapping(config_space_name)
    # rng_seed = cs.meta["rng_seed"] if folds_to_run_on is None \
    #     else cs.meta["seed_function_individual_fold"](cs.meta["rng_seed"], folds_to_run_on[0])
    rng_seed = seed

    config = Configuration(cs, name_grid_mapping[ensemble_method_name])
    cs.check_configuration(config)
    technique_run_args = get_ensemble_switch_case_config(config,
                                                         rng_seed=rng_seed, metric=ens_metric,
                                                         n_jobs=n_jobs,
                                                         is_binary=is_binary,
                                                         labels=list(range(mt.n_classes)))
    print("Run for Config:", config)

    # -- Run Evaluation
    print("#### Process Task {} for Dataset {} with Ensemble Technique {} ####".format(
        mt.openml_task_id,
        mt.dataset_name,
        ensemble_method_name))

    # -- Re-Add Custom Preprocessor
    from sklearn.compose import ColumnTransformer
    from sklearn.compose import make_column_selector
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    import numpy as np

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[("cat", enc, make_column_selector(dtype_include=["category", "object"]))],
        remainder="passthrough"
    )


    def compute_scores(ensemble_test_y, y_pred_ensemble_model):
        internal_scores = []
        # Compute all scores

        # 1. accuracy
        internal_scores.append(accuracy_score(ensemble_test_y, y_pred_ensemble_model))

        # 2. balanced accuracy
        internal_scores.append(balanced_accuracy_score(ensemble_test_y, y_pred_ensemble_model))

        # 3. f1 macro
        internal_scores.append(f1_score(ensemble_test_y, y_pred_ensemble_model, average="macro"))

        return np.array(internal_scores)


    print(ens_save_name)
    scores = evaluate_ensemble_on_metatask(mt, technique_name=ens_save_name, **technique_run_args,
                                           output_dir_path=out_path, store_results="parallel",
                                           save_evaluation_metadata=True,
                                           return_scores=compute_scores,
                                           folds_to_run=folds_to_run_on,
                                           use_validation_data_to_train_ensemble_techniques=True,
                                           verbose=True,
                                           isolate_ensemble_execution=isolate_execution,
                                           predict_method=predict_method,
                                           store_metadata_in_fake_base_model=True,
                                           preprocessor=preprocessor)
    print(scores)
    average_score = np.sum(scores, axis=0) / len(scores)
    print(f"K-Fold Average Performance: {average_score}")

    print("Storing State")
    s_path.touch()
    print("Done")

    # Save the results to a .txt file
    # timestamp = benchmark_name.split("_")
    # timestamp = timestamp[1] + "_" + timestamp[2]
    timestamp = "0"
    out_path = file_path.parent / "evaluation_custom"
    os.makedirs(out_path, exist_ok=True)
    out_path = out_path / f"{openml_task_id}_{seed}.txt"

    # - Append to .txt save file
    with open(out_path, "a") as f:
        f.write(ensemble_method_name + "_seed_" + str(seed) + ":\n")
        f.write(f"Score per fold: {scores}\n")
        f.write(f"Average Score: {average_score}\n")
        f.write("\n" + "#" * 100 + "\n" * 2)

    div_out = file_path.parent / "evaluation_custom" / f"{openml_task_id}_"
    try:
        if ensemble_method_name != "SingleBest":
            compute_diversity(mt, str(div_out), technique_run_args["technique"])
    except:
        print("Diversity Computation went wrong!")
