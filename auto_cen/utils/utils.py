"""
A module containing miscellaneous functions.
"""
from typing import Union

import numpy as np
import pandas as pd

def labels_to_numpy(labels: Union[pd.Series, pd.DataFrame]) -> np.array:
    """
    Converts labels of either type pandas Series or DataFrame to a numpy array.

    :param labels: The label(s).
              If binary/multiclass a pandas Series or a DataFrame with one column.
              If multilabel a pandas DataFrame where each column is one label.
    :return: A numpy array of either shape (n_samples,) or (n_samples, n_labels).
    """
    if isinstance(labels, np.ndarray):
        return labels

    # Check for multi-labels
    if isinstance(labels, pd.Series) or len(labels.columns) > 1:
        labels = labels.to_numpy()
    else:
        labels = labels.iloc[:, 0].to_numpy()
    return labels

def process_configuration(prefixes: list, algorithm: str, config: dict) -> dict:
    """
    Given a dictionary, split it up in the configurations of its included algorithms.

    :param prefixes: The prefixes used by the ConfigSpace configurations for the different algorithms.
    :param config: The combined configuration as a dict.
    :param algorithm: The classifier algorithm
    :return: The classification config. (dict), name of the diversity alg. (str), config. of
    the diversity alg. (dict)
    """
    processed_configs = {}
    for p in prefixes:
        new_config = {k[len(p) + 1:]: v for k, v in config.items() if k.startswith(p)}

        if p != algorithm:
            if not new_config:  # If the method (prefix) was not in the configuration, set it as empty
                processed_configs[p] = ('', {})
            elif p == "ENS":
                new_config = {k.split(':')[-1]: v for k, v in new_config.items()}
                processed_configs[p] = ('size', new_config)
            else:
                # For all non-classifiers another prefix has to be removed.
                algo = new_config.pop('algorithm')
                new_config = {k.split(':')[-1]: v for k, v in new_config.items()}
                processed_configs[p] = (algo, new_config)
        else:
            processed_configs[p] = (algorithm, new_config)
    return processed_configs


