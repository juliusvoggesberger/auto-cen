"""
Module that contains methods for reading in data.
"""

import pickle
from typing import Union

import pandas as pd

from pandas.core.dtypes.common import is_numeric_dtype
from sklearn import preprocessing

from auto_cen.constants import FILEPATH_MODELS
from auto_cen.main.ensemble import EnsembleLearner


class DataReader:
    """
       Class for reading in a dataset.
    """

    def __init__(self):
        self.unique_y = set()

    def read_csv(self, filepath: Union[str, list], class_col: str = 'class'):
        """
        Reads in the dataset if given as a .csv.

        :param filepath: If only a single dataset is read: Pass it as a string.
                         If multiple datasets (i.e. one for training, test, validation,...) are read:
                            They are passed as a list of strings.
        :param class_col: The name of the column which holds the class labels.

        :return: Either a tuple for a single dataset, where the entries are the data X and the labels y
                 Or a list of tuples for multiple datasets.
        """

        X, y, ds = [], [], []

        if isinstance(filepath, list):
            for file in filepath:
                data, labels = self._read_csv(file, class_col)
                X.append(data)
                y.append(labels)
            for i in range(len(y)):
                # Convert string labels to int labels
                # Do this here to prevent that one of the datasets has an incomplete set of classes
                # And as such its classes are incorrectly mapped
                y[i] = self._convert_string_to_int_labels(y[i], class_col)
                ds.append(X[i])
                ds.append(y[i])
            return ds
        else:
            X, y = self._read_csv(filepath, class_col)
            return X, self._convert_string_to_int_labels(y, class_col)

    def _read_csv(self, filepath: str, class_col: str):
        """
        Method for reading in a single csv file.

        :param filepath: The filepath to the dataset as a string.
        :param class_col: The name of the column which holds the class labels.
        :return: A tuple, where the entries are the data X and the labels y.
        """

        dframe = pd.read_csv(filepath, delimiter=",")
        y = dframe[[class_col]]
        X = dframe.drop(class_col, axis=1)
        self.unique_y.update(y["class"].unique().tolist())

        return X, y

    def _convert_string_to_int_labels(self, dframe: pd.DataFrame, class_col: str) -> pd.DataFrame:
        """
        Given a DataFrame, convert the values of the label columns to integers.

        :param dframe: The DataFrame.
        :param class_col: Column names of the label columns of the DataFrame.
        :return: The DataFrame, where the label column values are changed to integers.
        """
        if not is_numeric_dtype(dframe[class_col].dtype):
            enc = preprocessing.LabelEncoder()
            enc.fit(list(self.unique_y))
            dframe[class_col] = enc.transform(dframe[class_col])
        return dframe


def load_ensemble_model(filename: str) -> EnsembleLearner:
    """
    Load a pickled ensemble model from files/models/filename.

    :param filename: The filename of the pickle file.
    :return: The EnsembleLearner object
    """
    with open(FILEPATH_MODELS + filename, "rb") as file:
        ensemble = pickle.load(file)
        file.close()
    return ensemble
