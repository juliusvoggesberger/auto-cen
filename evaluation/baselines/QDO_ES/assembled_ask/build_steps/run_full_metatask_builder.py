import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "../.."))  # have to do this because of singularity. I hate it

from pathlib import Path
from shutil import rmtree
from assembled_ask.util.metatask_base import get_metatask
from assembled_ask.util.metric_switch_case import msc

from assembled_ask.ask_assembler import AskAssembler

if __name__ == "__main__":
    # -- Get Input Parameter
    openml_task_id = sys.argv[1]
    time_limit = int(sys.argv[2])
    memory_limit = int(sys.argv[3])
    metric_name = sys.argv[4]
    file_format = sys.argv[5]
    base_folder_name = sys.argv[6]
    filter_predictors = sys.argv[7]  # "TopN", "SiloTopN"
    seed = int(sys.argv[10])

    # New Parameters
    is_dataset = sys.argv[8] == "True"
    folds_to_run = [int(x) for x in sys.argv[9].split(",")] if "," in sys.argv[9] else [int(sys.argv[9])]
    num_folds = len(folds_to_run)

    if file_format not in ["csv", "hdf", "feather"]:
        raise ValueError("Unknown file format: {}".format(file_format))

    # -- Build paths
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    tmp_output_dir = file_path.parent.parent / "benchmark/output/{}/task_{}".format(base_folder_name, openml_task_id)
    print("Full Path Used: {}".format(tmp_output_dir))

    # -- Clean Previous results if they exist
    out_path = tmp_output_dir.joinpath("final_output/{}".format(filter_predictors))
    print("Clean Up Previous Results")
    if out_path.exists():
        rmtree(out_path)
    os.makedirs(out_path)

    # -- Rebuild The Metatask - either from OpenML Task or OpenML Dataset
    if is_dataset:
        print(f"Building Metatask for OpenML Dataset: {openml_task_id}")
        mt = get_metatask(openml_task_id, is_dataset, num_folds=num_folds, seed=seed)
    else:
        print("Building Metatask for OpenML Task: {}".format(openml_task_id))
        mt = get_metatask(openml_task_id)

    mt.file_format = file_format
    mt.save_chunk_size = 500

    # -- Init and run assembler
    print("Run Assembler")
    assembler = AskAssembler(mt, tmp_output_dir, folds_to_run=folds_to_run, resampling_strategy="cv")
    metric_to_optimize = msc(metric_name, len(mt.class_labels) == 2, list(range(mt.n_classes)))

    assembler.set_constraints(metric_to_optimize.name, time_limit, memory_limit, None, False, False)
    mt = assembler.build_metatask_from_predictor_data(pruner=filter_predictors,
                                                      metric=metric_to_optimize, seed=seed)

    if mt is not None:
        print(f"Saving Metatask to File: {out_path}")
        mt.to_files(out_path)
        print("Finished Saving to File")

    print("Saving State")
    s_path = file_path.parent.parent / "benchmark/state/{}/task_{}/".format(base_folder_name, openml_task_id)
    s_path.mkdir(parents=True, exist_ok=True)
    (s_path / "run_full_metatask_builder_{}.done".format(filter_predictors)).touch()
    print(f"Saved State to: {s_path}")
