from auto_cen.constants import BALANCED_ACCURACY, DOUBLEFAULT_NORM

BLUEPRINT = "nohup python run_auto_cen_benchmark.py {} {} {} {} {} {} {} {} {} {} {} > nohup_{}.out\n"

budget_m = 2400  # Time budget for classifiers: 45min
budget_f = 1200  # Time budget for decision fusion: 15min
n_splits = 10  # Number of cross-validation splits for the evaluation of classifiers and decision fusion
cutoff_time = 600  # Use ten minutes as a maximum runtime of model training
sel_heuristic = "None"
perf_metric = BALANCED_ACCURACY  # Use balanced accuracy for the performance metric
div_metric = DOUBLEFAULT_NORM # Use double fault as the diversity metric
ensemble_size = 20  # Use 20 as the maximum ensemble size
seeds = [123, 456, 789, 1010, 2020] # The random seed used in the evaluation

dataset_ids = [6, 184, 554, 1590, 1459, 40668, 40983, 41027, 40927, 40926, 40701, 1489, 41156,
               41168, 4538, 40996, 4135, 1461, 41166, 42769]

with open("commands.sh", "w") as f:
    f.write("#!/bin/bash \n")
    f.write("source venv/bin/activate \n")
    for seed in seeds:
        for i, data_id in enumerate(dataset_ids):
            output_file = str(data_id) + "_M" + str(budget_m) + "_F" + str(budget_f) + "_CV" + str(
                n_splits) + "_CO" + str(cutoff_time) + "_SH" + str(
                sel_heuristic) + "_PERF" + perf_metric + "_DIV" + div_metric + "_SIZE" + str(
                ensemble_size) + "_SEED" + str(seed) + "_AUTOMLBENCH"
            f.write(
                BLUEPRINT.format(data_id,
                                 budget_m,
                                 budget_f,
                                 cutoff_time,
                                 n_splits,
                                 sel_heuristic,
                                 perf_metric,
                                 div_metric,
                                 seed,
                                 ensemble_size,
                                 output_file))

            # Delete the psmac3 ouputs older than 15min to avoid building up too many folders
            if i > 0 and i % 3 == 0:
                f.write(
                    "find ~/auto_cen -name \"psmac3-*\" -type d -mmin +1200 -exec rm -rf {} +\n")
                f.write("\n")
