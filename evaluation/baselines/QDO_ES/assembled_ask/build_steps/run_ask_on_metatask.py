import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "../.."))  # have to do this because of singularity. I hate it

from pathlib import Path
from assembled_ask.util.metric_switch_case import msc
from assembled_ask.util.metatask_base import get_metatask
from assembled_ask.ask_assembler import AskAssembler

if __name__ == "__main__":
    # -- Get Input Parameter
    openml_task_id = sys.argv[1]
    time_limit = int(sys.argv[2])
    memory_limit = int(sys.argv[3])
    folds_to_run = [int(x) for x in sys.argv[4].split(",")] if "," in sys.argv[4] else [int(sys.argv[4])]
    metric_name = sys.argv[5]
    base_folder_name = sys.argv[6]

    # New Parameters
    is_dataset = sys.argv[7] == "True"
    num_folds = len(folds_to_run)
    n_jobs = int(sys.argv[8])
    seed = int(sys.argv[9])

    # -- Build paths
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    tmp_output_dir = file_path.parent.parent / "benchmark/output/{}/task_{}".format(base_folder_name, openml_task_id)
    print("Full Path Used: {}".format(tmp_output_dir))

    # -- Get The Metatask - either from OpenML Task or OpenML Dataset
    if is_dataset:
        print(f"Building Metatask for OpenML Dataset: {openml_task_id} with {num_folds} folds")
        mt = get_metatask(openml_task_id, is_dataset, num_folds, seed)
    else:
        print("Building Metatask for OpenML Task: {}".format(openml_task_id))
        mt = get_metatask(openml_task_id)

    metric_to_optimize = msc(metric_name, len(mt.class_labels) == 2, list(range(mt.n_classes)))

    # -- Init and run assembler
    print("Run Assembler")

    # Setting the seed here (for Auto-sklearn) is somehow not working (Don't know why or how to fix it)
    #TODO Resampling = cv
    assembler = AskAssembler(mt, tmp_output_dir, resampling_strategy="cv", folds_to_run=folds_to_run)
    # TODO max_models_on_disc=50
    assembler.run_ask(metric_to_optimize, time_limit, memory_limit, n_jobs=n_jobs, n_folds=num_folds, seed=seed, max_models_on_disc=50)

    print("Finished Run, Save State")
    for fold in folds_to_run:
        s_path = file_path.parent.parent / "benchmark/state/{}/task_{}/".format(base_folder_name, openml_task_id)
        s_path.mkdir(parents=True, exist_ok=True)
        (s_path / "run_ask_on_metatask_{}.done".format(fold)).touch()
