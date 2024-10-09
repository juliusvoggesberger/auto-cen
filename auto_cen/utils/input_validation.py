"""
Module used to create a specification for a given data set.
"""

from typing import Union
import pandas as pd
import pandas.api.types

from auto_cen.pipeline.ensemble_algorithms.base_method import BaseMethod
from auto_cen.constants import MULTILABEL, BINARY, MULTICLASS, MIXED, NUMERICAL, CATEGORICAL, \
    CONTINUOUS_OUT, LABELS


def generate_specification(X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> (list, dict):
    """
    Given some input data generate a specification and select the classifiers that fit the
    specification.
    The specification consists of the input data types, the classification problem and the possible
    output types.
    The input data is assumed to be given as a pandas DataFrame.
    Categorical data - of type string, int, bool, etc. - is assumed to have dtype categorical.
    Continuous data is assumed to be of dtype numerical.

    :param X: Data as a DataFrame of shape (n_samples, n_features)
    :param y: Labels for the data in X. Either a Series or a DataFrame.
              If a Series, it is of shape (n_samples,).
              If a DataFrame, it is of shape (n_samples, n_labels).
    :return: A list of classifiers, The specification as a dict with keys 'input','output','problem'
    """

    specification = {}

    # Check how many labels the data has. If more than 1, it is a multi-label problem.
    if isinstance(y, pd.Series):
        n_labels = 1
    elif isinstance(y, pd.DataFrame):
        n_labels = len(y.columns)
    else:
        raise TypeError(f"Given classification target is not of type {pd.DataFrame} or {pd.Series},"
                        f"but of type {type(y)}")  # Problem Specification
    multilabel = False
    if n_labels > 1:
        multilabel = True

    multiclass = False
    if isinstance(y, pd.DataFrame):
        for column in y:
            n_classes = len(y[column].unique())
            if n_classes > 2:
                multiclass = True
                break
    elif len(y.unique()) > 2:
        multiclass = True

    if multilabel and multiclass:
        raise ValueError("Multilabel + multiclass problems are not supported.")

    # Input Data
    is_numerical = False
    is_categorical = False
    if isinstance(y, pd.Series):
        if pandas.api.types.is_numeric_dtype(X.dtypes):
            is_numerical = True
        else:
            is_categorical = True
    else:
        for column in X:
            if pandas.api.types.is_numeric_dtype(X[column]):
                is_numerical = True
            else:
                is_categorical = True
            if is_numerical and is_categorical:
                # If both are True we do not have to check any further
                break

    specification['problem'] = _get_problem(multiclass, multilabel)
    specification['input'] = _get_input_type(is_numerical, is_categorical)

    classifier, output = get_classifier(specification)
    specification['output'] = output
    return classifier, specification


def get_classifier(specification: dict) -> (list, tuple):
    """
    Given a specification, consisting of a problem and input types,
    select the classifiers that conform to that specification.

    :param specification: A dict with two keys: 'problem' and 'input'.
    :return: A list of classifiers and a tuple containing the output types all classifiers support.
    """
    prob_output = True
    classifier = BaseMethod().get_classifier_for_spec(specification)
    for clf in classifier:
        if CONTINUOUS_OUT not in clf[1].get_specification_config()['output']:
            prob_output = False

    if prob_output:
        output = (LABELS, CONTINUOUS_OUT)
    else:
        output = (LABELS,)

    return classifier, output


def get_combiner(specification: dict) -> (list, list):
    """
    Given a specification, consisting of a problem and input types,
    select the combiners that conform to that specification.

    :param specification: A dict with three keys: 'problem', 'input' and 'output'.
    :return: A list of combiners with no hyperparameters, A list of combiners with hyperparameters.
    """
    combiner_simple, combiner_hp = BaseMethod().get_combiner_for_spec(specification)
    return combiner_simple, combiner_hp


def _get_problem(m_class: bool, m_label: bool) -> tuple:
    """
    Returns the problem specification.

    :param m_class: True, if multi-class.
    :param m_label: True, if multi-label.
    :return: A tuple containing the specification
    """
    if m_class and m_label:
        problem = (MULTICLASS, MULTILABEL)
    elif m_class:
        problem = (MULTICLASS,)
    elif m_label:
        problem = (MULTILABEL,)
    else:
        problem = (BINARY,)
    return problem


def _get_input_type(num: bool, cat: bool) -> str:
    """
    Returns the input types of the given data.
    Either categorical, continuous or mixed.

    :param num: True, if numerical features.
    :param cat: True, if categorical features
    :return: Input type as string.
    """
    if num and cat:
        input_type = MIXED
    elif num:
        input_type = NUMERICAL
    elif cat:
        input_type = CATEGORICAL
    else:
        raise ValueError("Unknown Input Type.")
    return input_type
