"""
Script to train machine learning model.
author: Philippe Jean Mith
Date: April 13th 2023
"""

# Add the necessary imports for the starter code.
import os
import pickle
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# Initialize logging
logging.basicConfig(filename='../logs/train_model.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

from ml.model import (
    train_model, 
    save_confusion_matrix,
    evaluation, 
    inference,
    slices_evaluation
)
from ml.data import process_data

# Add code to load in the data.
data_path = "../data/clean_census.csv"
data = pd.read_csv('../data/clean_census.csv')

# Create the OneHotEncoder and LabelBinarizer() objects
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
binarizer = LabelBinarizer()

logging.info("Split the data into train and test")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

logging.info("Start processing the train data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
logging.info(f"Processed train data: {X_train.shape}, {y_train.shape}")
logging.info("Start processing the test data")
# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, 
    label="salary", training=False, encoder=encoder, lb=binarizer
)
logging.info(f"Processed test data: {X_test.shape}, {y_test.shape}")
# Group the train data into a xgb.DMatrix
#test_xgb_matrix = xgb.DMatrix(X_test, label=y_test)

# Train and save a model.
xgb_model = train_model(X_train, y_train)
# Evaluate the model on the test data
logging.info("Start the evaluation of the model")
threshold = 0.52
# evaluate trained model on test set
predictions = inference(xgb_model, X_test)
recall, precision, fscore, accuracy, y_test, y_predict, predicted_data = evaluation(
                                                                            test, predictions,
                                                                              y_test, threshold)
logging.info(
    f"Test evaluation result: \
        \n Recall: {recall} \n Precision: {precision} \
        \n Fscore: {fscore} \n Accuracy: {accuracy} "
    )
cm = save_confusion_matrix(y_test, y_predict)
logging.info(f"Save the Confusion matrix:\n{cm}")
pickle.dump(xgb_model, open("../model/xgb_model.pkl", "wb"))
pickle.dump(encoder, open("../model/encoder.pkl", 'wb'))
pickle.dump(lb, open("../model/labelizer.pkl", 'wb'))
logging.info(f"Saved models: {os.listdir('../model')}")
predicted_data.to_csv("../data/predicted_data.csv", index=False)
logging.info(f"Predicted data save: {os.listdir('../data')}")

# Delete the slice prediction if it has been already created
slice_pred_file_path = "../data/slice_output.txt"
if os.path.exists(slice_pred_file_path):
    os.remove(slice_pred_file_path)
logging.info("Starting the slice evaluation")
for feature in enumerate(cat_features):
    performance_df = slices_evaluation(test, feature[1], y_test, np.array(y_predict))
    if feature[0] == 0:
        performance_df.to_csv(slice_pred_file_path,  mode='a', index=False)
    else:
        performance_df.to_csv(slice_pred_file_path,  mode='a', header=False, index=False)
logging.info(f"Predicted slice data save: {os.listdir('../data')}")