"""
Script for FastAPI instance and model inference
author: Philippe Jean Mith
Date: April 13th 2023
"""

# Put the code for your API here.
from fastapi import FastAPI, HTTPException
from typing import Union, Optional

# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel
import pandas as pd
import os, pickle
from starter.ml.data import process_data

# instantiate FastAPI app
app = FastAPI(
    title="Salary prediction API",
    description="An API that predict the salary of an employee base ",
    version="1.0.0",
)

# Declare the data object with its components and their type.
class InputData(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    
    
# Ouput for data validation
class Output(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    score: float
    prediction: str

@app.get("/")
async def greetings():
    # load the model from
    global model, encoder, lb
    model = pickle.load(open(os.path.join("model","xgb_model.pkl"), "rb"))
    encoder = pickle.load(open(os.path.join("model", "encoder.pkl"), "rb"))
    lb = pickle.load(open(os.path.join("model","labelizer.pkl"), "rb"))
    return "Welcome to the salary prediction API"


# This allows sending of data (our InferenceSample) via POST to the API.
@app.post("/predict/", response_model=Output)
async def ingest_data(inference: InputData):
    data = {  'age': inference.age,
                'workclass': inference.workclass, 
                'fnlgt': inference.fnlgt,
                'education': inference.education,
                'education-num': inference.education_num,
                'marital-status': inference.marital_status,
                'occupation': inference.occupation,
                'relationship': inference.relationship,
                'race': inference.race,
                'sex': inference.sex,
                'capital-gain': inference.capital_gain,
                'capital-loss': inference.capital_loss,
                'hours-per-week': inference.hours_per_week,
                'native-country': inference.native_country,
                }

    # prepare the sample for inference as a dataframe
    sample = pd.DataFrame(data, index=[0])

    # apply transformation to sample data
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
    
        
    sample,_,_,_ = process_data(
                                sample, 
                                categorical_features=cat_features, 
                                training=False, 
                                encoder=encoder, 
                                lb=lb
                                )

    # get model prediction which is a one-dim array like [1]                            
    prediction = model.predict(model, sample)
    score = prediction
    # convert prediction to label and add to data output
    if prediction[0]>0.5:
        data['prediction'] = '>50K'
        data['score'] = prediction[0]
    else:
        data['prediction'] = '<=50K'
        
    data['score'] = prediction[0]

    return data

if __name__ == '__main__':
    pass