
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

import preprocessors as pp
import config


# Creating pipeline for the data preprocessing steps
heart_failure_pipe1 = Pipeline(
    [
        ('log_transformer',
            pp.LogTransformer(variables=config.NUMERICALS_LOG_VARS)),

        ('scaler', MinMaxScaler()),
    ]
)

# Creating pipeline for the machine learning model
heart_failure_pipe2 = Pipeline(
    [
        
        ('randomforest_classification', RandomForestClassifier(n_estimators=390, max_features = "auto", min_samples_leaf = 10, random_state = 11850))
        
    ]
)

