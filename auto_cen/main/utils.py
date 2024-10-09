"""
Module that contains different functions that are used in the main package.
"""
from typing import List

from auto_cen.main.data_storage import DataStore
from auto_cen.optimization.history import RunInfo
from auto_cen.pipeline.base import BaseAlgorithm
from auto_cen.pipeline.generic_clf_pipeline import ClassifierPipeline
from auto_cen.pipeline.generic_cmb_pipeline import CombinerPipeline


def retrain_ensemble(data: DataStore, models: List[RunInfo], fusion: RunInfo) -> (
        List[BaseAlgorithm], BaseAlgorithm):
    """
    Retrain the ensemble for given model and fusion configurations.

    :param data: The object holding the necessary data.
    :param models: The List of RunInfo objects holding the classifier configurations
    :param fusion: The RunInfo object holding the fusion configuration
    :return: List of trained classifier models, Trained fusion model
    """

    # Retrain the k-models of the ensemble
    retrained = []
    for model in models:
        clf = ClassifierPipeline(model.algorithm, model.configuration, model.rnd_seed, data.cat_indices)
        clf.fit(data.X_train, data.y_train)
        retrained.append(clf)
    fm = CombinerPipeline(fusion.algorithm, fusion.configuration, seed=fusion.rnd_seed)
    X_fm, y_fm, _, _ = data.compute_fusion_data(models)
    fm.fit(X_fm, y_fm)
    return retrained, fm
