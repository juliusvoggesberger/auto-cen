import numpy as np
import pandas as pd

from auto_cen.optimization.history import RunInfo


def create_table(title: str, columns: list, sections: list) -> str:
    """
    Creates a table as a string.

    :param title: Title of the table.
    :param columns: List of the column names.
    :param sections: The content of the tables as a list.

    :return: The table as a string.
    """
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


def save_csv(rows: list, header: list, filename:str):
    """
    Save results in a csv file.

    :param rows: The data as rows.
    :param header: The header of the csv file.
    :param filename: The name of the csv file.
    """
    data_frame = pd.DataFrame(data=rows, columns=header)
    data_frame.to_csv(filename)


def model_as_row(model: RunInfo, metrics: dict, prefix:list=None) -> list:
    """
    Converts a model (i.e. classification/decision fusion model) into a list which can be used
    as a row in a table, either for saving in a csv file or for printing in a string.

    :param model: The RunInfo of the model.
    :param metrics: A dictionary of the evaluation metrics.
                    Keys are the metric names, values their evaluation results.
    :param prefix: Some additional information describing the evaluation result,
                   e.g. its evaluation name.

    :return: A list containing the model and evaluation metrics.
    """
    if prefix is None:
        prefix = []
    return prefix + [model.rid, model.algorithm, model.configuration, model.runtime,
                     *metrics.values()]

def preprocess_metrics(metrics: dict) -> list:
    """
    Transforms a dictionary of evaluation results into a list and rounds them to 3 decimals.

    :param metrics: A dictionary of the evaluation metrics.

    :return: A list containing the rounded evaluation metrics.
    """
    return [round(m, 3) if not isinstance(m, np.ndarray) else str(np.round(m, 3)[:2]) + ".."
            for m in metrics.values()]
