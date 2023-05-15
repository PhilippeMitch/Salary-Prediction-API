"""
Script for FastAPI instance and model inference
author: Philippe Jean Mith
Date: April 13th 2023
"""

# Put the code for your API here.
from fastapi import FastAPI, HTTPException
from typing import Any
import asyncio

# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel
from starter.ml.model import inference
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
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
                        "example": {
                                    'age':50,
                                    'workclass':"Private", 
                                    'fnlgt':234721,
                                    'education':"Doctorate",
                                    'marital_status':"Separated",
                                    'occupation':"Exec-managerial",
                                    'relationship':"Not-in-family",
                                    'race':"Black",
                                    'sex':"Female",
                                    'capital_gain':0,
                                    'capital_loss':0,
                                    'hours_per_week':50,
                                    'native_country':"United-States"
                                    }
                        }


# Load the models when the application start
@app.on_event("startup")
async def startup_event(): 
    global model, encoder, lb
    # Try to load the models
    try:
        model = pickle.load(open(os.path.join('model','xgb_model.pkl'), "rb"))
        encoder = pickle.load(open(os.path.join('model','encoder.pkl'), "rb"))
        lb = pickle.load(open(os.path.join('model','labelizer.pkl'), "rb"))
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": err},
        )

@app.get("/")
async def greetings():
    # Greeting for the home page
    return "Welcome to the salary prediction API"


# This allows sending of data (our InferenceSample) via POST to the API.
@app.post("/predict/")
async def ingest_data(predict: InputData) -> Any:
    data = {  'age': predict.age,
                'workclass': predict.workclass, 
                'fnlgt': predict.fnlgt,
                'education': predict.education,
                'marital-status': predict.marital_status,
                'occupation': predict.occupation,
                'relationship': predict.relationship,
                'race': predict.race,
                'sex': predict.sex,
                'capital-gain': predict.capital_gain,
                'capital-loss': predict.capital_loss,
                'hours-per-week': predict.hours_per_week,
                'native-country': predict.native_country,
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
    # Run the startup_event 
    await asyncio.gather(startup_event())
    sample,_,_,_ = process_data(
                                sample, 
                                categorical_features=cat_features, 
                                training=False, 
                                encoder=encoder, 
                                lb=lb
                                )

    # Get the prediction
    prediction = inference(model, sample)
    # convert prediction to label and add to data output
    if prediction[0].item()>0.5:
        data['prediction'] = '>50K'
        data['score'] = round(prediction[0].item(),2)
    else:
        data['prediction'] = '<=50K'
        
    data['score'] = round(prediction[0].item(),2)

    return data

if __name__ == '__main__':
    pass