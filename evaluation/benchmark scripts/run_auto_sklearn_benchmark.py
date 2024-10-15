import sys

from csv import writer
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from openml import datasets
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from auto_cen.constants import BALANCED_ACCURACY, DOUBLEFAULT_NORM, DOUBLEFAULT

from ConfigSpace import ConfigurationSpace # from ConfigSpace.configuration_space

from auto_cen.utils.diversity_metrics import get_oracle_predictions, compute_diversity_metrics
from auto_cen.utils.performance_metrics import get_performance_report
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import INPUT, SPARSE, DENSE, UNSIGNED_DATA
import autosklearn.pipeline.components.data_preprocessing
from autosklearn.askl_typing import FEAT_TYPE_TYPE

from sklearn.preprocessing import OneHotEncoder

class OHE(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """
        This preprocessors does not change the data
        Only One-hot Encoding is applied
        """
        # Some internal checks makes sure parameters are set
        for key, val in kwargs.items():
            setattr(self, key, val)
        categories = [i for i in range(len(self.feat_type)) if list(self.feat_type.values())[i] == 'categorical' ]
        enc = OneHotEncoder(handle_unknown="ignore",sparse=False)
        self.model = ColumnTransformer(transformers=[("cat", enc, categories)], remainder="passthrough")

    def fit(self, X, Y=None):
        self.model.fit(X)
        return self

    def transform(self, X):
        X_t = self.model.transform(X)
        return X_t

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "OHE",
            "name": "OneHotEncoding",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
            feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()  # Return an empty configuration as there is None

def create_table(title: str, columns: list, sections: list):
    blueprint = "{:<20}" * len(columns)
    header_divider = "=" * (18 * (len(columns) + 1)) + "\n"
    row_divider = "-" * (18 * (len(columns) + 1)) + "\n"

    # Title and header
    output_table = title + "\n" + header_divider
    output_table += blueprint.format(*columns) + "\n" + row_divider

    # Contents
    for section in sections:
        # Section Header
        if section[0] != "":
            output_table += section[0] + "\n"
        for row in section[1]:
            output_table += blueprint.format(*row) + "\n"
        output_table += row_divider
    return output_table

def save_csv(rows, header, filename):
    data_frame = pd.DataFrame(data=rows, columns=header)
    data_frame.to_csv(filename)


def _get_n_classes_and_uniques(y) -> (int, list):
    all_classes = set()
    y_unique = np.unique(y).tolist()
    all_classes.update(y_unique)
    all_classes.discard(None)
    return len(all_classes), list(all_classes)


def evaluate(predictions, y_true, n_classes, labels):
    metrics = [BALANCED_ACCURACY]
    report = get_performance_report(y_true, predictions, None, metrics, n_classes, labels)

    return report


def evaluate_diversity(ensemble, X_train, y_train, X_test, y_true, filename, encoder):
    metrics = [DOUBLEFAULT_NORM, DOUBLEFAULT]
    ensemble_diversity = np.zeros(len(metrics))
    div_pairs = []
    y_test_transf = encoder.transform(y_true)

    ensemble.refit(X_train, y_train)
    models = ensemble.get_models_with_weights()
    # clf_fitted = [[0] for clf in models]
    clf_pred = np.array([clf[1].predict(X_test) for clf in models])
    diff = [get_oracle_predictions(pred, y_test_transf) for pred in clf_pred]
    pairs_idx = list(combinations(range(0, len(models)), 2))

    for pair in pairs_idx:
        idx1, idx2 = pair
        divs = compute_diversity_metrics(diff[idx1], diff[idx2], clf_pred[idx1], clf_pred[idx2],
                                         y_test_transf, metrics)
        ensemble_diversity += list(divs.values())
        div_pairs.append(
            [models[idx1][1].config.get_dictionary(), models[idx2][1].config.get_dictionary(),
             divs])

    ensemble_diversity = np.around(ensemble_diversity / len(pairs_idx), 3)

    header = ["Algorithm Pair"] + metrics
    ensemble_row = ["", [["Ensemble", *ensemble_diversity]]]
    clf_rows = ["", [(pair[0]['classifier:__choice__'] + " + " + pair[1]['classifier:__choice__'],
                      *np.around(list(pair[2].values()), 3)) for pair in div_pairs]]
    print(create_table("Ensemble Diversity", header, [ensemble_row, clf_rows]))

    header = ["m1 algorithm", "m1 configuration", "m2 algorithm", "m2 configuration"] + metrics
    rows = [
        [pair[0]['classifier:__choice__'], pair[0], pair[1]['classifier:__choice__'], pair[1],
         *list(pair[2].values())]
        for pair in div_pairs]
    save_csv(rows, header, filename + "_diversity" + str(seed) + ".csv")


print("Setup Autosklearn NoPreprocessing")
autosklearn.pipeline.components.data_preprocessing.add_preprocessor(OHE)

if __name__ == '__main__':
    dataset_id = int(sys.argv[1])
    budget = int(sys.argv[2])
    seed = int(sys.argv[3])

    memory_limit = None

    # Run for 1h
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=budget,
                                                              initial_configurations_via_metalearning=0,
                                                              resampling_strategy="cv",
                                                              resampling_strategy_arguments={
                                                                  "folds": 10},
                                                              tmp_folder="tmp_" + str(
                                                                  dataset_id) + "/",
                                                              seed=seed, per_run_time_limit=600,
                                                              metric=autosklearn.metrics.balanced_accuracy,
                                                              delete_tmp_folder_after_terminate=False,
                                                              n_jobs=8, include={
                                                              'feature_preprocessor': ["no_preprocessing"], "data_preprocessor": ["OHE"]}, memory_limit=memory_limit)
    dataset = datasets.get_dataset(dataset_id, download_data=True, download_qualities=False)
    X, y, _, _ = dataset.get_data(dataset.default_target_attribute)
    n_classes, labels = _get_n_classes_and_uniques(y)

    enc = preprocessing.LabelEncoder()
    enc.fit(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=seed)


    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)

    report_dict = evaluate(y_hat, y_test, n_classes, labels)
    print("Accuracy score", report_dict)

    with open('autosklearn_report' + str(seed) +'.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow([str(dataset_id)] + list(report_dict.values()))
        f_object.close()

    enc = preprocessing.LabelEncoder()
    enc.fit(y)
    evaluate_diversity(automl, X_train, y_train, X_test, y_test, "autosklearn_" + str(dataset_id),
                       enc, seed)



