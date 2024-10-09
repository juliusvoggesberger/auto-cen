"""
Module that contains classes for storing data.
"""

import math
from typing import Union, List

import pandas as pd
import numpy as np

from pandas.core.dtypes.common import is_numeric_dtype
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from auto_cen.optimization.history import RunInfo
from auto_cen.pusion.utils import transform_label_tensor_to_class_assignment_tensor, \
    transform_label_vector_to_class_assignment_matrix
from auto_cen.utils.utils import labels_to_numpy


class DataStore:
    """
    Stores Dataset data as Training, Validation and Test Data for machine learning models.
    If needed can generate Validation and Test Data.

    :param X_train: Training data
    :param y_train: Training labels
    :param X_valid: Validation data
    :param y_valid: Validation labels
    :param X_test: Test data
    :param y_test: Test labels
    :param train_size: Fraction of the given data used as the training set.
    :param valid_size: Fraction of the given data used as the validation set.
    :param test_size: Fraction of the given data used as the test set.
    :param stratify: Used for stratified sampling if not None. Only used when train/valid split
                     is executed.
    :param seed: Random seed.
    """

    def __init__(self, X_train: Union[pd.DataFrame, np.ndarray],
                 y_train: Union[pd.DataFrame, np.ndarray],
                 X_valid: Union[pd.DataFrame, np.ndarray] = None,
                 y_valid: Union[pd.DataFrame, np.ndarray] = None,
                 X_test: Union[pd.DataFrame, np.ndarray] = None,
                 y_test: Union[pd.DataFrame, np.ndarray] = None,
                 train_size: float = 0.6, valid_size: float = 0.3, test_size: float = 0.1,
                 stratify: list = None, n_splits: int = 1, seed: int = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test

        self.feat_types = self.X_train.dtypes if isinstance(X_train, pd.DataFrame) else None

        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size

        self.stratify = stratify
        self.n_splits = n_splits
        self.seed = seed

        self.n_classes, self.y_unique = self._get_n_classes_and_uniques()
        self.cat_mask, self.cat_indices = self._get_cat_mask()
        self.enc = None

        self.prepare_data()

    def convert_labels(self):

        self.enc = preprocessing.LabelEncoder()
        self.enc.fit(self.y_unique)
        self.y_train = self.enc.transform(self.y_train)
        if self.y_valid is not None:
            self.y_valid = self.enc.transform(self.y_valid)
        if self.y_test is not None:
            self.y_test = self.enc.transform(self.y_test)

    def prepare_data(self):
        """
        Given data as DataFrames, prepare it, so it can be used by the framework.
        This means ensuring that all data sets (train, valid, test) are given and converted to numpy
        If they are not given, create them.
        """
        if not math.isclose(1.0, self.train_size + self.valid_size + self.test_size, abs_tol=0.001):
            raise ValueError("Data set fractions do not sum up to one")

        if self.n_splits > 1 and self.valid_size > 0:
            raise ValueError(
                "When cross-validation is used no validation data will be used. Set valid_size to 0.")

        if self.test_data_is_none():
            self.X_train, self.y_train, self.X_test, self.y_test = \
                self.split_data(self.X_train, self.y_train,
                                train_size=self.train_size + self.valid_size,
                                valid_size=self.test_size, stratify=self.stratify)
        else:
            # noinspection PyTypeChecker
            self.X_test, self.y_test = self._data_to_numpy(self.X_test, self.y_test)

        # Set train and valid data and convert to numpy
        if self.valid_data_is_none() and self.valid_size > 0:

            train_size = self.train_size * (1.0 / (1 - self.test_size))
            stratify = self.y_train if self.stratify is not None else None

            self.X_train, self.y_train, self.X_valid, self.y_valid = \
                self.split_data(self.X_train, self.y_train, train_size=train_size,
                                valid_size=1 - train_size, stratify=stratify)
        elif self.valid_data_is_none():
            # If validation size is given as 0, use training data as validation
            self.X_train, self.y_train = self._data_to_numpy(self.X_train, self.y_train)
            self.X_valid, self.y_valid = self.X_train, self.y_train
        else:
            self.X_train, self.y_train = self._data_to_numpy(self.X_train, self.y_train)
            self.X_valid, self.y_valid = self._data_to_numpy(self.X_valid, self.y_valid)

        self.convert_labels()

    def split_data(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray],
                   train_size: float, valid_size: float, stratify: list = None) -> (
            np.array, np.array, np.array, np.array):
        """
        Splits given data into two parts: train and validation.
        The train-/valid-sizes have to add up to one.

        :param X: A DataFrame holding the data, has shape (n_samples, n_features).
        :param y: A DataFrame holding the labels that fit to the data, has shape
                  (n_samples,n_labels).
        :param train_size: Fraction of the data used for the training set.
        :param valid_size: Fraction of the data used for the validation set.
        :param stratify: Used for stratified sampling if not None.
        :return: Four arrays. X_train, y_train, X_valid, y_valid
        """
        if train_size + valid_size != 1.0:
            raise ValueError("Train-and Validation size do not sum to one.")

        if not isinstance(y, np.ndarray):
            y = labels_to_numpy(y)

        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size,
                                                              stratify=stratify,
                                                              random_state=self.seed)
        return X_train, y_train, X_valid, y_valid

    def compute_fusion_data(self, models: List[RunInfo]):
        """
        Prepare the data needed by the combiner.

        :param models: The models used to compute the fusion data as a list of RunInfo objects.
        :return: - Model output predictions for validation data (n_classifier, n_samples, n_classes)
                 - The true validation labels (n_samples, n_classes)
                 - Model output predictions for test data (n_classifier, n_samples, n_classes)
                 - The true test labels (n_samples, n_classes)
        """
        y_valid_pred = np.asarray([m.prediction_va for m in models])
        y_test_pred = np.asarray([m.prediction_ts for m in models])

        # y_valid_tensor: True labels, y_valid_ensemble_out: Predictions
        y_valid_tensor, y_valid_ensemble_out = self._prepare_data(y_valid_pred, self.y_valid)
        # y_test_tensor: True labels, y_test_ensemble_out: Predictions
        y_test_tensor, y_test_ensemble_out = None, None
        if y_test_pred.size > 0:
            # Only need this if no CV is used, i.e. if a test prediction is made
            y_test_tensor, y_test_ensemble_out = self._prepare_data(y_test_pred, self.y_test)

        return y_valid_ensemble_out, y_valid_tensor, y_test_ensemble_out, y_test_tensor

    def _prepare_data(self, y_pred: np.array, y_true: np.array) -> (np.array, np.array):
        """
        Prepare the data needed by the combiner.

        :param y_pred: Model predictions.
        :param y_true: The true labels.
        :return: 1. y: A matrix of form (n_samples, n_classes) -> The true training labels
                 2. y_ens: A tensor of form (n_classifier, n_samples, n_classes) -
                           > The training predictions
        """
        # First get the model predictions
        y_ens = y_pred
        if len(y_pred.shape) < 3:  # If not probability predictions
            y_ens = transform_label_tensor_to_class_assignment_tensor(y_pred, self.n_classes)
        y_true = transform_label_vector_to_class_assignment_matrix(y_true, self.n_classes)

        return y_true, y_ens

    def _get_n_classes_and_uniques(self) -> (int, list):
        all_classes = set()
        y_unique = np.unique(self.y_train).tolist() + np.unique(
            self.y_valid).tolist() + np.unique(self.y_test).tolist()
        all_classes.update(y_unique)
        all_classes.discard(None)
        return len(all_classes), list(all_classes)

    def _get_cat_mask(self) -> (list, list):
        mask = []
        int_mask = []

        for i, col in enumerate(self.X_train.columns):
            if not is_numeric_dtype(self.X_train[col].dtype):
                mask.append(col)
                int_mask.append(i)
        return mask, int_mask

    def _convert_string_to_int_labels(self, y: np.array) -> pd.DataFrame:
        """
        Given a DataFrame, convert the values of the label columns to integers.

        :param y: The labels
        :return: The DataFrame, where the label column values are changed to integers.
        """

        if not is_numeric_dtype(y.dtype):
            enc = preprocessing.LabelEncoder()
            enc.fit(self.y_unique)
            y = enc.transform(y)
        return y

    def valid_data_is_none(self) -> bool:
        return self.X_valid is None or self.y_valid is None

    def test_data_is_none(self) -> bool:
        return self.X_test is None or self.y_test is None

    @staticmethod
    def _data_to_numpy(X, y):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y = labels_to_numpy(y) if y is not None else None
        return X, y

    def clean_up(self):
        """
        Remove all data instances to save memory, but keep the metadata (i.e. label encoder, n_classes, etc.)
        """

        self.X_train, self.X_test, self.X_test = None, None, None
        self.y_train, self.y_test, self.y_test = None, None, None
        self.stratify = None
