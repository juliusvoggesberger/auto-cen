BLUEPRINT = "nohup python run_divbo_benchmark.py --datasets {} --task_type cls --algos bo_div --rep_num 1 --iter_num 50 --ens_size 25 > nohup_{}.out\n"


dataset_ids = [6, 184, 554, 1590, 1459, 40668, 40983, 41027, 40927, 40926, 40701, 1489, 41156,
               41168, 4538, 40996, 4135, 1461, 41166, 42769]

with open("commands.sh", "w") as f:
    f.write("#!/bin/bash \n")
    f.write("source venv/bin/activate \n")
    for j in range(5): # Create 5 runs
        for i, data_id in enumerate(dataset_ids):
            output_file = "divbo_" + str(data_id) + "_run" + str(j)
            f.write(BLUEPRINT.format(data_id, output_file))
