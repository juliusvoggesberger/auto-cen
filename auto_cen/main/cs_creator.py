"""
Module consisting function for creating the configuration space.
"""
from typing import Union, Tuple

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, \
    Constant

from auto_cen.constants import ENCODER
from auto_cen.pipeline.transformations.pre_processing.encoding import Encoder


def create_cs(algorithms: list, ens_size: Union[int, Tuple] = None, has_cat: bool = False,
              default_value: str = None) -> ConfigurationSpace:
    """
    Creates a configuration space given a list of algorithms using the SMAC config space
    representation.

    :param algorithms: List of tuples containing the algorithm names and the algorithm models.
    :param ens_size: If this parameter is a tuple, this will be used to define the interval
                     in which the ensemble size will be searched.
                     Else if it is an int, the ensemble size will be added as a constant.
                     If None, nothing will be done.
    :param has_cat: True, if any of the features are categorical.
                            Used to decide if an encoder is needed.

    :return: List of algorithms with configuration space.
    """
    c_space = _create_cs(algorithms, default_value)

    # Add preprocessing search spaces (preprocessing, feature engineering, implicit diversity)
    subspaces = _add_preprocessors(has_cat)
    for space in subspaces:
        if space[1] is not None:
            c_space.add_configuration_space(*space)

    # Ensemble size as part of the search space
    if isinstance(ens_size, Tuple):
        ens_size = UniformIntegerHyperparameter('ENS:size', lower=ens_size[0],
                                                upper=ens_size[1])
        c_space.add_hyperparameter(ens_size)
    # Fixed Ensemble size
    elif ens_size is not None:
        ens_size = Constant('ENS:size', value=ens_size)
        c_space.add_hyperparameter(ens_size)
    return c_space


def _add_preprocessors(has_cat: bool) -> list:
    """
    Creates subspaces for preprocessing methods.
    Currently only encoding is supported.

    :param has_cat: True, if any of the features are categorical.
                            Used to decide if an encoder is needed.
    :return: List of subspaces.
    """
    implicit_div = [("NODIV", None)]
    preprocessors = [("NOPRE", None)]
    feature_engineers = [("NOFE", None)]
    enc_cs = [("NOEnc", None)]
    if has_cat:
        enc_cs = [(ENCODER, Encoder)]

    return [("Div", _create_cs(implicit_div)), ("Pre", _create_cs(preprocessors)),
            ("Fe", _create_cs(feature_engineers)), ("Enc", _create_cs(enc_cs))]


def _create_cs(algorithms: list, default_value: str = None) -> ConfigurationSpace:
    """
    Creates a configuration space given a list of algorithms using the SMAC config space
    representation.

    :param algorithms: List of tuples containing the algorithm name and the algorithm model.
    :return: List of algorithms with configuration space.
    """
    if not algorithms:
        return None

    c_space = ConfigurationSpace()
    root = CategoricalHyperparameter('algorithm', [a[0] for a in algorithms],
                                     default_value=default_value)
    c_space.add_hyperparameter(root)
    for algo in algorithms:
        if algo[1] is not None:
            # Needed, if a root node without an algorithm is passed.
            # This is the case, if e.g. it should be possible that no diversity or preprocessing
            # method should be used.
            c_space.add_configuration_space(algo[0], algo[1].get_config_space(),
                                            parent_hyperparameter={'parent': root,
                                                                   'value': algo[0]})
    return c_space
