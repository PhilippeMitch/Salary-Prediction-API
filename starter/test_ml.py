"""
Unit test of model.py module with pytest
author: Philippe Jean Mith
Date: April 13th 2023
"""

import pandas as pd
import pytest, os, logging, pickle
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import inference

@pytest.fixture(scope="module")
def data():
    # Load in the data.
    data_path = "data/clean_census.csv"
    return pd.read_csv(data_path)

@pytest.fixture(scope="module")
def features(data):
    """
    Return the categorical features as argument
    """
    cat_features = list(data.select_dtypes(['object', 'category']).columns)[:-1]
    return cat_features

@pytest.fixture(scope="module")
def model():
    """
    Check if the saved model exist
    """
    model_path = "model/xgb_model.pkl"
    if os.path.isfile(model_path):
        model = pickle.load(open(model_path, 'rb'))
    else:
        logging.error(
            f"Model {model_path} can't found")
    return model

@pytest.fixture(scope="module")
def train_dataset(data, features):
    """
    Returns cleaned train dataset
    """
    train, _ = train_test_split( data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )
    X_train, y_train, _, _ = process_data(
                                            train,
                                            categorical_features=features,
                                            label="salary",
                                            training=True
                                        )
    return X_train, y_train

def test_import_data():
    """
    Test presence and shape of dataset file
    """
    data_path = "data/clean_census.csv"
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the df shape
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    
def test_processed_data(train_dataset):
    """
    Check if the number of feature are the same
    """
    X_train, y_train = train_dataset

    try:
        assert X_train.shape[0] == len(y_train)
        logging.info('Testing test_processed_data: SUCCESS')
    except AssertionError as err:
        logging.error(
        f"The size of the features are different from the target: \
            {X_train.shape[0]} {len(y_train)}")
        raise err

def test_inference(model, train_dataset):
    """
    Check inference function
    """
    X_train, y_train = train_dataset

    try:
        preds = inference(model, X_train)
        assert len(y_train) == len(preds)
        logging.info('Testing test_inference: SUCCESS')
    except Exception as err:
        logging.error(
        "Could not performe inference with the saved model and train data")
        raise err

def test_slice_prediction():
    file_path = "data/sclice_output.txt"
    if os.path.isfile(file_path):
        try:
            slice_preds = pd.read_csv(file_path)
            assert slice_preds.shape[0] > 0
            assert slice_preds.shape[1] > 0
            logging.info('Testing test_slice_prediction: SUCCESS')
        except AssertionError as err:
            logging.error(
            f"could not verify the {file_path}")
            raise err
    else:
        logging.error(
            f"Could not find the file {file_path}")