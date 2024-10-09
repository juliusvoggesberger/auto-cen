"""
Module that contains functions that write to disk.
"""

import pickle
from pathlib import Path

import numpy as np

from auto_cen.constants import FILEPATH_MODELS
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.utils.utils import process_configuration


def save_models(models: list, filename: str):
    """
    Pickles a list of models and saves it to files/models/filename

    :param models: list of models.
    :param filename: filename of the pickled history.
    """
    Path(FILEPATH_MODELS).mkdir(parents=True, exist_ok=True)
    with open(FILEPATH_MODELS + filename, "wb") as file:
        pickle.dump(models, file, -1)


def print_evaluation(fusion_method: BaseAlgorithm, fusion_metrics: dict, models_eval: list,
                     models: list, evaluation_performance_metrics: list):
    """
    Prints the evaluation of the Ensemble

    :param fusion_method: Fusion method of the ensemble
    :param fusion_metrics: Performance metric values of the ensemble
    :param models_eval: Performance metric values of the models
    :param models: The models
    :param evaluation_performance_metrics: The metrics that were used for the evaluation
    """
    print("Evaluation Results:")
    print("=" * 40 * (len(evaluation_performance_metrics) + 1))
    # generate string header
    empty_methods = ["", "", "", ""]
    blueprint = "{:<20}"*5 + "{:<20}" * len(evaluation_performance_metrics)
    print(blueprint.format("Algorithm", "Encoder", "PreProc", "FE", "Diversity", *evaluation_performance_metrics))
    print("-" * 40 * (len(evaluation_performance_metrics) + 1))
    print("Combiner")
    print(blueprint.format(fusion_method.get_model().get_specification_config()['name'], *empty_methods,
                           *[round(m, 3) if not isinstance(m, np.ndarray) else "" for m in
                             fusion_metrics.values()]))
    print("-" * 40 * (len(evaluation_performance_metrics) + 1))
    print("Classifiers")
    for i, model in enumerate(models):
        prefixes = ['Enc', 'Pre', 'Fe', 'Div', model[1].algorithm]
        configurations = process_configuration(prefixes, model[1].algorithm, model[1].configuration)
        clf = [configurations.pop(model[1].algorithm)[0]]
        for k, v in configurations.items():
            clf.append(v[0])
        print(
            blueprint.format(*clf, *[round(m, 3) if not isinstance(m, np.ndarray) else "" for m in
                                    models_eval[i][1].values()]))
