BLUEPRINT = "nohup python run_auto_sklearn_benchmark.py {} {} {} > nohup_{}.out\n"

budget = 3600 # # Time budget for auto-sklearn: 60min
seeds = [123, 456, 789, 1010, 2020] # The random seed used in the evaluation

dataset_ids = [6, 184, 554, 1590, 1459, 40668, 40983, 41027, 40927, 40926, 40701, 1489, 41156,
               41168, 4538, 40996, 4135, 1461, 41166, 42769]

with open("commands.sh", "w") as f:
    f.write("#!/bin/bash \n")
    f.write("source venv/bin/activate \n")
    for seed in seeds:
        for i, data_id in enumerate(dataset_ids):
            output_file = "autosklearn_" + str(data_id) + "_M" + str(budget) + str(seed)
            f.write(BLUEPRINT.format(data_id, budget, seed, output_file))

            # Delete auto-sklearn outputs to avoid filling up the memory
            if i > 0 and i % 2 == 0:
                f.write(
                    "find ~/auto_sklearn -name \"tmp_*\" -type d -mmin +60 -exec rm -rf {} +\n")
                f.write("\n")
        f.write("\n")
    f.write("find ~/auto_sklearn -name \"tmp_*\" -type d -mmin +60 -exec rm -rf {} +\n")
