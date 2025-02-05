import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "../.."))  # have to do this because of singularity. I hate it

from pathlib import Path
from assembled_ask.util.metatask_base import get_metatask
from assembled_ask.ask_assembler import AskAssembler

if __name__ == "__main__":
    # -- Get Input Parameter
    openml_task_id = sys.argv[1]
    folds_to_run = [int(x) for x in sys.argv[2].split(",")] if "," in sys.argv[2] else [int(sys.argv[2])]
    metric_name = sys.argv[3]
    base_folder_name = sys.argv[4]
    refit = sys.argv[5] == "refit" #TODO NEW
    save_disc_space = sys.argv[6] == "save_space"

    # New Parameters
    is_dataset = sys.argv[7] == "True"
    seed = int(sys.argv[8])
    num_folds = len(folds_to_run)

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

    # -- Init and run assembler
    print("Run Assembler")
    assembler = AskAssembler(mt, tmp_output_dir, folds_to_run=folds_to_run, save_disc_space=save_disc_space, resampling_strategy="cv")
    assembler.collect_predictor_data_from_ask_data(refit=refit, seed=seed)
    print("Finished Collecting Predictor Data")

    print("Save State")
    for fold in folds_to_run:
        s_path = file_path.parent.parent / "benchmark/state/{}/task_{}/".format(base_folder_name, openml_task_id)
        (s_path / "run_collect_predictor_data_{}.done".format(fold)).touch()

    print(f"Finished Saving State to: {file_path.parent.parent / 'benchmark/state/{}/task_{}/'.format(base_folder_name, openml_task_id)}")
