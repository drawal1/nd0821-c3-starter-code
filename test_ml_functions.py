"""
    author: Dhar Rawal
    date: 12/18/2022
    pytest tests for churn_library.py
"""

import pandas as pd

from starter.starter.ml.data import process_data
import starter.starter.ml.model as model

def data_sample_and_cat_features(purpose):
    """ return data file path and list of cat features
        purpose: Pass 0 for encoding, 1 for training and 2 for inference
    """
    data_file_path = './starter/data/census.csv'
    data = pd.read_csv(data_file_path)
    assert data is not None

    if purpose == 0:
        sample_size = int(data.shape[0]*1.0)
    elif purpose == 1:
        sample_size = int(data.shape[0]*0.8)
    elif purpose == 2:
        sample_size = 5
    else:
        raise AttributeError("Invalid value passed for purpose")

    data_sample = data.sample(sample_size)

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

    return (data_sample, cat_features)

def test_process_data_for_training():
    '''
    test the process data function
    '''
    # first get the encoder and lbl_bnrzr
    data_sample, cat_features = data_sample_and_cat_features(0)
    _, _, encoder, lbl_bnrzr = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=True
    )

    # get training data
    data_sample, cat_features = data_sample_and_cat_features(1)
    x_train, y_train, _, _ = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lbl_bnrzr
    )

    assert x_train is not None
    assert type(x_train).__name__ == "ndarray"
    assert y_train is not None
    assert type(y_train).__name__ == "ndarray"
    assert encoder is not None
    assert type(encoder).__name__ == "OneHotEncoder"
    assert lbl_bnrzr is not None
    assert type(lbl_bnrzr).__name__ == "LabelBinarizer"

def test_process_data_for_inference():
    '''
    test the process data function
    '''
    # first get the encoder and lbl_bnrzr
    data_sample, cat_features = data_sample_and_cat_features(0)
    _, _, encoder, lbl_bnrzr = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=True
    )

    # get inference data sample
    data_sample, _ = data_sample_and_cat_features(2)
    x_inf, y_inf, _, _ = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lbl_bnrzr
    )

    assert x_inf.shape[0] == 5
    assert y_inf.shape == (5,)

def test_train_model():
    '''
    test the train_model function
    '''
    # first get the encoder and lbl_bnrzr
    data_sample, cat_features = data_sample_and_cat_features(0)
    _, _, encoder, lbl_bnrzr = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=True
    )

    # get training data
    data_sample, cat_features = data_sample_and_cat_features(1)
    x_train, y_train, _, _ = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lbl_bnrzr
    )

    lrc_model = model.train_model(x_train, y_train)

    assert lrc_model is not None
    assert type(lrc_model).__name__ == 'LogisticRegression'

def test_inference():
    '''
    test the inference function
    '''
    # first get the encoder and lbl_bnrzr
    data_sample, cat_features = data_sample_and_cat_features(0)
    _, _, encoder, lbl_bnrzr = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=True
    )

    # get training data
    data_sample, cat_features = data_sample_and_cat_features(1)
    x_train, y_train, _, _ = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lbl_bnrzr
    )

    lrc_model = model.train_model(x_train, y_train)

    data_sample, _ = data_sample_and_cat_features(2)
    x_inf, _, _, _ = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lbl_bnrzr
    )

    y_pred = model.inference(lrc_model, x_inf)

    assert y_pred is not None
    assert type(y_pred).__name__ == "ndarray"
    assert y_pred.shape == (5,)

def test_compute_model_metrics():
    '''
    test the compute_model_metrics function
    '''
    # first get the encoder and lbl_bnrzr
    data_sample, cat_features = data_sample_and_cat_features(0)
    _, _, encoder, lbl_bnrzr = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=True
    )

    # get training data
    data_sample, cat_features = data_sample_and_cat_features(1)
    x_train, y_train, _, _ = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lbl_bnrzr
    )

    lrc_model = model.train_model(x_train, y_train)

    data_sample, _ = data_sample_and_cat_features(2)
    x_inf, y_inf, _, _ = process_data(
        data_sample, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lbl_bnrzr
    )

    y_pred = model.inference(lrc_model, x_inf)

    precision, recall, fbeta = model.compute_model_metrics(y_inf, y_pred)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None

    print(f"precision: {precision}, recall: {recall}, fbeta: {fbeta}")
