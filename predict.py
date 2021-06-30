
import pandas as pd

import joblib
import config


def make_prediction(input_data):
    
    _pipe_heart_failure1 = joblib.load(filename=config.PIPELINE1_NAME)
    _pipe_heart_failure2 = joblib.load(filename=config.PIPELINE2_NAME)
    
    input_data = _pipe_heart_failure1.transform(input_data)
    results = _pipe_heart_failure2.predict(input_data)

    return results
   
if __name__ == '__main__':
    
    # test pipeline
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.metrics import roc_auc_score

    data_test = pd.read_csv(config.TEST_DATA_FILE)
    X_test = data_test[config.FEATURES]
    y_test = data_test[config.TARGET]

# Predicting the target values for the test set    
    y_pred = make_prediction(X_test)
    
    # determine classification_report and roc_auc_score for the test set

    print()
    print('----------------------------------------------------------------')
    print('Accuracy for the Test Set: {}'.format(accuracy_score(y_test, y_pred)))
    print()
    print("Classification report for the Test Set:")
    print()
    print(classification_report(y_test, y_pred)) 
    print("roc_auc_score for the Test Set: {}".format(roc_auc_score(y_test, y_pred)))
    print()

