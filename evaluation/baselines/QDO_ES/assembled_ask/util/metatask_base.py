import numpy as np
from sklearn.model_selection import PredefinedSplit, StratifiedKFold, StratifiedShuffleSplit
from assembled.metatask import MetaTask
from assembledopenml.openml_assembler import init_dataset_from_task

import openml
import os


def get_example_manual_metatask_for_ask() -> MetaTask:
    print("Get Toy Metatask")
    from sklearn.datasets import load_breast_cancer
    metatask_id = -1
    task_data = load_breast_cancer(as_frame=True)
    target_name = task_data.target.name
    dataset_frame = task_data.frame
    class_labels = np.array([str(x) for x in task_data.target_names])  # cast labels to string
    feature_names = task_data.feature_names
    cat_feature_names = []
    dataset_frame[target_name] = class_labels[dataset_frame[target_name].to_numpy()]
    metatask = MetaTask()
    fold_indicators = np.empty(len(dataset_frame))
    cv_spliter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for fold_idx, (train_index, test_index) in enumerate(
            cv_spliter.split(dataset_frame[feature_names], dataset_frame[target_name])):
        # This indicates the test subset for fold with number fold_idx
        fold_indicators[test_index] = fold_idx

    metatask.init_dataset_information(dataset_frame, target_name=target_name, class_labels=class_labels,
                                      feature_names=feature_names, cat_feature_names=cat_feature_names,
                                      task_type="classification", openml_task_id=metatask_id,
                                      dataset_name="breast_cancer.csv", folds_indicator=fold_indicators)
    return metatask


def get_openml_metatask_for_ask(mt_id) -> MetaTask:
    print("Get OpenML Metatask")

    metatask = MetaTask(file_format="feather")
    init_dataset_from_task(metatask, mt_id)
    metatask.read_randomness("OpenML", 0)

    return metatask

def get_openml_metatask_data_for_ask(mt_id, num_folds, data_folder='data/', seed: int = 0) -> MetaTask:
    """
    Get the metatask for an OpenML dataset. If the dataset has already been downloaded, it will be loaded from the
    cache. Otherwise, it will be downloaded and saved to the cache.

    :param mt_id: The OpenML dataset id
    :param num_folds: The number of folds to create for the dataset
    :param data_folder: The folder to store the dataset in
    :param seed: The seed to use
    """
    print("Get OpenML Metatask from data")

    metatask = MetaTask(file_format="feather")
    init_dataset_from_dataset_id(metatask, mt_id, num_folds, data_folder, seed)
    metatask.read_randomness("OpenML", seed)

    return metatask

def get_resampling_strategy(X: np.array, y: np.array, test_size: float = 0.2, seed: int = 0):
    """Get the resampling strategy appropriate for our framework and auto-sklearn."""

    fold_indicator = np.full(len(X), -1)

    cv = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    for fold_idx, (_, val_i) in enumerate(cv.split(np.zeros(len(y)), y)):
        fold_indicator[val_i] = fold_idx

    # return PredefinedSplit(fold_indicator)
    return fold_indicator

def create_folds_indicator(dataset: np.array, target_name: str, num_folds: int, seed: int = 0) -> np.array:
    """
    Create the fold indicators for the dataset. Only needed to handle the case of num_folds=1.

    :param dataset: The dataset to create the fold indicators for
    :param target_name: The name of the target column
    :param num_folds: The number of folds to create
    :param seed: The seed to use
    """
    fold_indicators = np.empty(len(dataset))

    if num_folds == 1:
        return get_resampling_strategy(dataset.drop(target_name, axis=1), dataset[target_name], seed=seed)

    else:
        # CV doesn't work in this framework with only one fold...
        cv_spliter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold_idx, (train_index, test_index) in enumerate(
                cv_spliter.split(dataset.drop(target_name, axis=1), dataset[target_name])):
            # This indicates the test subset for fold with number fold_idx
            fold_indicators[test_index] = fold_idx

        return fold_indicators

def init_dataset_from_dataset_id(meta_task, dataset_id, num_folds=10, data_folder='data/', seed: int = 0):
    """
    Custom method to load a dataset from OpenML into a Metatask using the dataset id.

    :param meta_task: The metatask object to fill with the dataset information
    :param dataset_id: The OpenML dataset id
    :param num_folds: The number of folds to create for the dataset
    :param data_folder: The folder to store the dataset in
    :param seed: The seed to use
    """
    if isinstance(dataset_id, str):
        dataset_id = int(dataset_id)

    # Ensure the data folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    metatask_file_path = os.path.join(data_folder, f"metatask_{dataset_id}.json")
    cached_task = os.path.exists(metatask_file_path)

    # Check if the dataset has already been downloaded
    if cached_task:
        print(f"Loading cached metatask from {metatask_file_path}")
        meta_task.read_metatask_from_files(data_folder, dataset_id)

        # Check if the number of folds matches the requested number of folds
        if len(np.unique(meta_task.folds)) != num_folds:
            print(f"WARNING: Number of folds in cached metatask ({len(np.unique(meta_task.folds))}) "
                  f"does not match the requested number of folds ({num_folds}).")
            print("Updated the metatask.")

            dataset = meta_task.dataset
            target_name = meta_task.target_name

            fold_indicators = create_folds_indicator(dataset, target_name, num_folds, seed)
            meta_task.folds = fold_indicators

        return meta_task
    else:
        print(f"Downloading dataset {dataset_id}")
        openml_dataset = openml.datasets.get_dataset(dataset_id)

    dataset_name = openml_dataset.name
    dataset, _, cat_indicator, feature_names = openml_dataset.get_data()
    target_name = openml_dataset.default_target_attribute
    class_labels = openml_dataset.retrieve_class_labels(target_name)

    # - Get Cat feature names
    cat_feature_names = [f_name for cat_i, f_name in zip(cat_indicator, feature_names) if
                         (cat_i == 1) and (f_name != target_name)]
    feature_names.remove(target_name)  # Remove only afterward, as indicator includes class

    # -- Check Task type
    if class_labels is not None:
        task_type = "classification"
    else:
        raise ValueError("Regression tasks are not supported yet.")

    # - Handle Folds like in get_example_manual_metatask_for_ask():
    # num_folds = 2 needs to be set here, for the case of num_folds=1 (Don't ask me why)
    folds_indicator = create_folds_indicator(dataset, target_name, 2, seed)

    # -- Fill object with values
    meta_task.init_dataset_information(dataset, target_name, class_labels, feature_names, cat_feature_names,
                                       task_type, dataset_id, folds_indicator, dataset_name)

    # Now set the correct number of folds
    if num_folds != 2:
        # - Handle Folds like in get_example_manual_metatask_for_ask():
        folds_indicator = create_folds_indicator(dataset, target_name, num_folds, seed)
        meta_task.folds = folds_indicator

    # -- Dump metatask
    if not cached_task:
        meta_task.to_files(data_folder)

    return meta_task


def get_metatask(openml_task_id, is_dataset: bool = False, num_folds: int = 10, seed: int = 0):
    """
    Get the metatask for an OpenML task or dataset.

    :param openml_task_id: The OpenML task or dataset id
    :param is_dataset: If True, the id is a dataset id, else it is a task id
    :param num_folds: The number of folds to create for the dataset
    """
    if is_dataset:
        mt = get_openml_metatask_data_for_ask(openml_task_id, num_folds, seed=seed)
    else:
        # Check if the task id is -1, in which case we use a toy example
        if openml_task_id == "-1":
            mt = get_example_manual_metatask_for_ask()
        else:
            mt = get_openml_metatask_for_ask(openml_task_id)

    return mt
