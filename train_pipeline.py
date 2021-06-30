
import numpy as np
import pandas as pd

import joblib

import pipeline
import config
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score


def run_training():
    """Train the model."""

    # read training data
    data_train = pd.read_csv(config.TRAINING_DATA_FILE)
    data_test = pd.read_csv(config.TEST_DATA_FILE)
    X_train = data_train[config.FEATURES]
    
    y_train = data_train[config.TARGET]
    X_test = data_test[config.FEATURES]
    y_test = data_test[config.TARGET]
 
# Data Preprocessing    
    pipeline.heart_failure_pipe1.fit(X_train, y_train)
 
# Saving the pipeline of Data Preprocessing steps    
    joblib.dump(pipeline.heart_failure_pipe1, config.PIPELINE1_NAME)
    
    print("Before Resampling of training data: ")
    counter = Counter(y_train)
    print(counter)
    
    _pipe_heart_failure1 = joblib.load(filename=config.PIPELINE1_NAME)
    
    data_train = pd.read_csv(config.TRAINING_DATA_FILE)
    data_test = pd.read_csv(config.TEST_DATA_FILE)
    X_train = data_train[config.FEATURES]
    y_train = data_train[config.TARGET]
    X_test = data_test[config.FEATURES]
    y_test = data_test[config.TARGET]
     
    X_train = _pipe_heart_failure1.fit_transform(X_train)
 
# Applying hybrid resampling (combination of over sampling and under sampling) on the training dataset    
    over = SMOTE(sampling_strategy=0.8)
    under = RandomUnderSampler(sampling_strategy=0.8)
    steps = [('o', over), ('u', under)]
    pipeline1 = Pipeline(steps=steps)	
    X_train, y_train = pipeline1.fit_resample(X_train, y_train)
    
    print("After Resampling of training data: ")
    counter = Counter(y_train)
    print(counter)
    
    pipeline.heart_failure_pipe2.fit(X_train, y_train)

# Saving the pipeline of Model    
    joblib.dump(pipeline.heart_failure_pipe2, config.PIPELINE2_NAME)

# Predicting the target values of training set    
    y_pred_train = pipeline.heart_failure_pipe2.predict(X_train)
    
# determine classification_report and roc_auc_score for the training set 
    
    print()
    print('---------------------------------------------------------')
    print('Accuracy for the Training Set is {}'.format(accuracy_score(y_train, y_pred_train)))
    print()
    print("Classification report for the Training Set:")
    print()
    print(classification_report(y_train, y_pred_train))
    print("roc_auc_score for the Training Set: {}".format(roc_auc_score(y_train, y_pred_train)))
    print()
    
    print("Training is completed")
    

if __name__ == '__main__':
    run_training()

