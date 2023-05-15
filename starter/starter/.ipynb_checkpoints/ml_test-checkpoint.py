"""
Unit test of model.py module with pytest
author: Philippe Jean Mith
Date: April 13th 2023
"""

import pandas as pd
import pytest, os, logging, pickle
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.data import process_data
from ml.model import inference, evaluation

@pytest.fixture(scope="module")
def data(data_path):
    # code to load in the data.
    return pd.read_csv(data_path)

@pytest.fixture(scope="module")
def features(data):
    """
    Fixture - will return the categorical features as argument
    """
    cat_features = list(data.select_dtypes(['object', 'category']).columns)[:-1]
    return cat_features

@pytest.fixture(scope="module")
def model():
    model_path = "./model/xgb_model.pkl"
    if os.path.isfile(model_path):
        model = pickle.load(open(model_path, 'rb'))
    else:
        logging.error(
            f"Model {model_path} can't found")
    return model

@pytest.fixture(scope="module")
def lb_and_encoder():
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    binarizer = LabelBinarizer()
    return (encoder, binarizer)

@pytest.fixture(scope="module")
def train_dataset(data, features):
    """
    Fixture - returns cleaned train dataset to be used for model testing
    """
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=features,
                                            label="salary",
                                            training=True
                                        )
    return X_train, y_train

@pytest.fixture(scope="module")
def test_dataset(data, features, lb_and_encoder):
    """
    Fixture - returns cleaned train dataset to be used for model testing
    """
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=10, 
                                stratify=data['salary']
                                )
    X_test, y_test, encoder, lb = process_data(
                                            train,
                                            categorical_features=features,
                                            label="salary",
                                            training=False,
                                            encoder=lb_and_encoder[0], 
                                            lb=lb_and_encoder[1]
                                        )
    return  X_test, y_test, test

"""
Test methods
"""
def test_import_data(data_path):
    """
    Test presence and shape of dataset file
    """
    try:
        data = pd.read_csv(data_path)
        return data
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    
def testing_processed_data(test_dataset, train_dataset):
    """
    Check if the number of feature are the same
    """
    X_train, y_train = train_dataset
    X_test, y_test, df_test = test_dataset

    try:
        assert X_train.shape[1] == X_test.shape[1]
    except AssertionError as err:
        logging.error(
        "The processed train and test data does not have the same amount of features")
        raise err

@pytest.fixture(scope="module") 
def test_inference(test_dataset):
    """
    Check inference function
    """
    X_test, y_test, df_test = test_dataset

    model_path = "./model/xgb_model.pkl"
    if os.path.isfile(model_path):
        model = pickle.load(open(model_path, 'rb'))

        try:
            preds = inference(model, X_test)
            assert len(y_test) == len(preds)
            logging('Testin test_inference: SUCCESS')
        except Exception as err:
            logging.error(
            "Inference cannot be performed on saved model and train data")
            raise err
    else:
        logging.error(
            f"Test test_inference: Model {model_path} can't found")
    return preds
        
def test_evaluation(data, model, inference):
    """
    Check the evaluation function
    """
    X_test, y_test, test_df = test_dataset
    predicted_data_path = "./data/predicted_data.csv"

    if os.path.isfile(predicted_data_path):
        predicted_data = pd.read_csv(predicted_data_path)

        try:
            recall, precision, fscore, accuracy, y_test, y_predict, predicted_data = evaluation(
                                                                            test_df, inference,
                                                                              y_test, 0.5)
            assert len(y_test) == len(y_predict)
            logging('Testin test_evaluation: SUCCESS')
        except Exception as err:
            logging.error(
            "Test test_evaluation: Evaluation cannot be performed on saved model and train data")
            raise err
    else:
        logging.error(
            f"Model {model_path} can't found")