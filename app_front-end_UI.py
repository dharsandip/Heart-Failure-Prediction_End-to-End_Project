
# Serve model as a flask application

import predict
import joblib
import numpy as np
from flask import Flask, request
import pandas as pd
import config
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

def make_prediction(input_data):
    
    _pipe_heart_failure1 = joblib.load(filename=config.PIPELINE1_NAME)
    _pipe_heart_failure2 = joblib.load(filename=config.PIPELINE2_NAME)
    
    input_data = _pipe_heart_failure1.transform(input_data)
    results = _pipe_heart_failure2.predict(input_data)

    return results
    
@app.route('/')
def home_endpoint():
    return 'Hello, Welcome to the Heart Failure Prediction App'

@app.route('/predict', methods=['POST'])
def get_prediction():
    
    """Let's predict the chance of death due to heart failure
    This is using docstrings for specification.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    
    responses:
        200:
            description: The output values
    """
    
    df_test=pd.read_csv(request.files.get("file"))
#    print(df_test.head())
    X_test = df_test[config.FEATURES]

    prediction = make_prediction(X_test)

    return 'Predictions of death: {}'.format(str(list(prediction)))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
