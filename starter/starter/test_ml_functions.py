"""
    author: Dhar Rawal
    date: 12/18/2022
    pytest tests for churn_library.py
"""

import sys
import pytest

import pandas as pd
from sklearn.model_selection import train_test_split

from .ml.data import process_data
from .ml import model
from .ml import slice as sl


class Tee(object):
    """ utility class to Tee console output to files """
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        """ write to files """
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        """ flush content """
        for f in self.files:
            f.flush()


@pytest.fixture
def data_and_cat_features():
    """return data file path and list of cat features"""
    data_file_path = "./starter/data/census.csv"
    data = pd.read_csv(data_file_path)
    assert data is not None

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    return (data, cat_features)


def test_process_data_for_encoder_lb(data_and_cat_features):
    """
    test the process data function for encoder and lb
    """
    # get the encoder and lb
    data = data_and_cat_features[0]
    cat_features = data_and_cat_features[1]
    _, _, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    assert encoder is not None
    assert type(encoder).__name__ == "OneHotEncoder"
    assert lb is not None
    assert type(lb).__name__ == "LabelBinarizer"


def test_process_data_for_train(data_and_cat_features):
    """
    test the process data function for training data
    """
    # get the encoder and lb
    data = data_and_cat_features[0]
    cat_features = data_and_cat_features[1]
    _, _, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    train, _ = train_test_split(data, test_size=0.20, stratify=data["salary"])

    # Process the training data with the encoder and lb
    x_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert x_train is not None
    assert type(x_train).__name__ == "ndarray"
    assert y_train is not None
    assert type(y_train).__name__ == "ndarray"


def test_train_model(data_and_cat_features):
    """
    test the train_model function
    """
    # get the encoder and lb
    data = data_and_cat_features[0]
    cat_features = data_and_cat_features[1]
    _, _, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    train, _ = train_test_split(data, test_size=0.20, stratify=data["salary"])

    # Process the training data with the encoder and lb
    x_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    lrc_model = model.train_model(x_train, y_train)

    assert lrc_model is not None
    assert type(lrc_model).__name__ == "LogisticRegression"


def test_inference(data_and_cat_features):
    """
    test the inference function
    """
    # get the encoder and lb
    data = data_and_cat_features[0]
    cat_features = data_and_cat_features[1]
    _, _, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    train, test = train_test_split(data, test_size=0.20, stratify=data["salary"])

    # Process the training data with the encoder and lb
    x_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    lrc_model = model.train_model(x_train, y_train)

    x_inf, _, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_pred = model.inference(lrc_model, x_inf)

    assert y_pred is not None
    assert type(y_pred).__name__ == "ndarray"


def test_compute_model_metrics(data_and_cat_features):
    """
    test the compute_model_metrics function
    """
    # get the encoder and lb
    data = data_and_cat_features[0]
    cat_features = data_and_cat_features[1]
    _, _, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    train, test = train_test_split(data, test_size=0.20, stratify=data["salary"])

    # Process the training data with the encoder and lb
    x_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    lrc_model = model.train_model(x_train, y_train)

    x_inf, y_inf, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_pred = model.inference(lrc_model, x_inf)

    precision, recall, fbeta = model.compute_model_metrics(y_inf, y_pred)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None

    print(f"precision: {precision:6.2n}, recall: {recall:6.2n}, fbeta: {fbeta:6.2n}")


def test_slice_performance(data_and_cat_features):
    """
    test the compute_model_metrics function
    """
    # get the encoder and lb
    data = data_and_cat_features[0]
    cat_features = data_and_cat_features[1]
    _, _, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    train, _ = train_test_split(data, test_size=0.20, stratify=data["salary"])

    # Process the training data with the encoder and lb
    x_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    lrc_model = model.train_model(x_train, y_train)

    original_stdout = sys.stdout # Save a reference to the original standard output
    with open('slice_output.txt', 'w', encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)

        for slicing_cat in cat_features:
            print(f"Slicing category: {slicing_cat}")
            print("| Categorical Value | Precision | Recall | fbeta |")
            print("| ----------------- | --------- | ------ | ----- |")
            slice_perf_dict = sl.slice_performance(
                data, cat_features, lrc_model, encoder, lb, slicing_cat
            )
            for key, val in slice_perf_dict.items():
                print(f"| {key:30s}| {val[0]:6.2n} | {val[1]:6.2n} | {val[2]:6.2n} |")

        #use the original
        sys.stdout = original_stdout
