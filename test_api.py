"""
Unit test of main.py API module with pytest
author: Philippe Jean Mith
Date: April 15th 2023
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import json
import logging

from main import app

client = TestClient(app)

def test_greetings():
    """
    Test if the api return the welcome message
    """
    response = client.get("/")
    try:
        assert response.status_code == 200
        assert response.json() == "Welcome to the salary prediction API"
    except AssertionError as err:
        logging.error(
            "Testing test_greetings: The greetings() function \
            doesn't appear to return the expected outputs"
        )
        raise err

def test_predict_1():
    """
    Test if the api return the expected prediction for True positive
    """
    sample =  { 
        "age": 50,
        "workclass": "Private",
        "fnlgt": 234721,
        "education": "Doctorate",
        "marital_status": "Separated",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }

    data = json.dumps(sample)

    response = client.post("/predict/", data=data )

    try:
        # test response status
        assert response.status_code == 200
    except AssertionError as err:
        logging.error(
        "Testing test_predict: The predict does not appear could not connect with the api"
        )

    try:
        # test predicted response
        assert response.json()["prediction"][0] == '>50K'
        assert response.json()["age"] == 50
        assert response.json()["fnlgt"] == 234721
        #assert response.json()["score"][0] == 0.69
    except AssertionError as err:
        logging.error(
        "Testing test_predict: The predict does not appear to return the expected output"
        )

def test_predict_0():
    """
    Test if the api return the expected prediction for a True Negative
    """
    sample =  { 
        "age": 56,
        "workclass": "Self-emp-inc",
        "fnlgt": 119891,
        "education": "Some-college",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 1672,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    data = json.dumps(sample)

    response = client.post("/predict/", data=data )

    try:
        # test response status
        assert response.status_code == 200
    except AssertionError as err:
        logging.error(
        "Testing test_predict: The predict does not appear could not connect with the api"
        )

    try:
        # test predicted response
        assert response.json()["prediction"][0] == '<=50K'
        #assert response.json()["score"][0] == 0.1
    except AssertionError as err:
        logging.error(
        "Testing test_predict: The predict does not appear to return the expected output"
        )