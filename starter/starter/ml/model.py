import logging
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.metrics import (
    fbeta_score,
    accuracy_score,
    f1_score,
    recall_score, 
    precision_score,
    confusion_matrix, 
    classification_report
)
from sklearn.metrics import precision_recall_fscore_support as score
sns.set()

# Initialize logging
# logging.basicConfig(filename='../starter/logs/train_model.log',
#                     level=logging.INFO,
#                     filemode='w',
#                     format='%(name)s - %(levelname)s - %(message)s')

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    def bo_tune_xgb(max_depth, gamma, eta):
        """
        Function with the internals we wish to maximize

        Inputs
        ------
        max_depth : tuple
                Range of maximum depth of a tree.
        gamma : tuple
            Range of minimum loss reduction required to make a further 
            partition on a leaf node of the tree.
        eta : tuple
            Range of step size shrinkage used in update to prevents overfitting. 

        """
        # Define the value range for the parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': int(max_depth),
            'eta': eta,
            'gamma': gamma,
            'eval_metric': 'aucpr'
        }

        #Cross validating with the specified parameters in 5 folds and 70 iterations
        cv_result = xgb.cv(params, training_xgb_matrix, num_boost_round=70, nfold=5)
        #Return the resul
        cv_result = cv_result['train-aucpr-mean'].iloc[-1]
        return 1.0 * cv_result
    
    # Group the train data into a xgb.DMatrix
    training_xgb_matrix = xgb.DMatrix(X_train, label=y_train)
    # Instantiate a BayesianOptimization
    xgb_bo = BayesianOptimization(
        bo_tune_xgb, {
            'max_depth': (3, 7),
            'gamma': (0, 1),
            'eta': (0.01, 0.4)
        }
    )
    
    logging.info("Start the hyperparameter tunning")
    # Run the the optimization
    xgb_bo.maximize(n_iter=6, init_points=8)
    # Show the best hypermarameters
    params = xgb_bo.max['params']
    logging.info(f"Hyperparameter tunning done: {params}")
    # Retrain the model with the best hyperparameter
    params = {
        'objective': 'binary:logistic',
        'eta': round(params['eta'], 1),
        'max_depth': round(params['max_depth']),
        'gamma': round(params['gamma']),
        'eval_metric': 'aucpr',
        'verbosity': 0,
    }

    # Create a list of xgb.DMatrix
    watch_list = [
                    (training_xgb_matrix, 'train')
                ]
    
    logging.info(f"Start training the model with the params: {params}")
    # Train the model with the selected hyperparameters
    xgb_model = xgb.train(params,
                              training_xgb_matrix,
                              num_boost_round=999,
                              evals=watch_list,
                              early_stopping_rounds=20)
    return xgb_model

def save_confusion_matrix(y_test, y_predicted):
    """
    Save the confusion matrix
    
    Inputs:
    -------
    y_predicted : numpy.ndarray
        The predictions result
    y_test : numpy.ndarray
        The actual label
    Returns:
    cm: str
        The confusion matrix
    """
    plt.figure(figsize=(15, 8))
    cm = confusion_matrix(y_test, y_predicted)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_title('XGBoost Confusion Matrix', fontsize=20)
    ax.set_xlabel('\nPredicted Values')
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticklabels(['0', '1'], fontsize = 20)
    ax.xaxis.tick_top()
    ax.set_ylabel('Actual Values')
    ax.yaxis.set_ticklabels(['0', '1'], fontsize = 15)
    plt.savefig('../screenshots/confusion_matrix.png')
    return cm

def classify_type(y_pred, y_label):
    """
    Get classification type
    
    Inputs:
    -------
    y_pred : numpy.ndarray
        The predictions result
    y_label : numpy.ndarray
        The actual label
    Returns:
    --------
    rs : str
        The clasification type
    """
    
    rs = 'TP' if y_pred == 1 and y_label == 1 else 'FP' if y_pred == 1 and y_label == 0 else 'TN' if y_pred == 0 and y_label == 0 else 'FN'
    return rs

def classify_type(y_pred, y_label):
    """
    Get classification type
    
    Inputs:
    -------
    y_pred : numpy.ndarray
        The predictions result
    y_label : numpy.ndarray
        The actual label
    Returns:
    --------
    rs : str
        The clasification type
    """
    
    rs = 'TP' if y_pred == 1 and y_label == 1 else 'FP' if y_pred == 1 and y_label == 0 else 'TN' if y_pred == 0 and y_label == 0 else 'FN'
    return rs

def evaluation(data, predictions, y_test,threshold):
    """
    Get the evaluation result
    Inputs:
    -------
     y_predict : numpy.ndarray
        The predictions result
    y_test : numpy.ndarray
        The actual label
    Returns:
    --------
    precision : float
            The model precision
    recall : float
        The recall ratio
    fscore : float
        The F1 score
    accuracy : float
        The accuracy of the model
    y_test: numpy.ndarray
        The test label
    y_predict : numpy.ndarray
            The predicted label
    """
    data['label'] = y_test
    data['predicted_score'] = predictions
    data['predicted_label'] = data.apply(lambda x: 1 if x['predicted_score'] >= threshold else 0, axis = 1)
    data['type'] = data.apply(lambda x: classify_type(x['predicted_label'], x['label']), axis = 1)
    y_predict = data['predicted_label'].tolist()
    precision, recall, fscore, support = score(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    cl_report = classification_report(y_test, data['predicted_label'].values, target_names=['0', '1'])
    logging.info(f"Classification report:\n{cl_report}")

    return round(recall[1], 2), round(precision[1], 2), \
            round(fscore[1], 2), round(accuracy, 2), \
                y_test, y_predict, data


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    fscore : fscore
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    fscore = f1_score(y, preds, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta, fscore


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    X_matrix = xgb.DMatrix(X)
    preds = model.predict(X_matrix)
    
    return preds

def slices_evaluation(data, feature, y_data, y_preds):
    """
    Function that outputs the performance of the model on slices of the data.
    ------
    data: Dataframe
        Data on which we will perform the evaluation by slice
    feature: str
        Feature on which to perform the slices
    y_data : np.array
        Slice data labels
    y_preds : np.array
        Predicted labels

    Returns
    ------
    slice_eval_df: Dataframe
        A dataframe that contains the evaluation metrics for each slice
    """    
    slice_options = list(data[feature].unique())
    slice_eval_df = pd.DataFrame(
        index=slice_options, 
        columns=['feature','n_samples','precision', 'recall', 'fbeta', 'fscore']
    )
    for option in slice_options:
        slice_mask = data[feature]==option

        slice_y = y_data[slice_mask]
        slice_preds = y_preds[slice_mask]
        precision, recall, fbeta, fscore = compute_model_metrics(slice_y, slice_preds)
        
        slice_eval_df.loc[option, 'feature'] = feature
        slice_eval_df.loc[option, 'n_samples'] = len(slice_y)
        slice_eval_df.loc[option, 'precision'] = precision
        slice_eval_df.loc[option, 'recall'] = recall
        slice_eval_df.loc[option, 'fbeta'] = fbeta
        slice_eval_df.loc[option, 'fscore'] = fscore

    slice_eval_df.reset_index(names='feature value', inplace=True)
    list_column = list(slice_eval_df.columns)
    list_column[0], list_column[1] =  list_column[1], list_column[0]
    slice_eval_df = slice_eval_df[list_column]

    return slice_eval_df
