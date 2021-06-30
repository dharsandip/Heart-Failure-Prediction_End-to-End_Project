This is a complete machine learning end-to-end pipeline project where
our goal is not only making predictions with good accuracy but also
productionizing the enire machine learning pipeline, creating Web API with front end UI, creating 
docker image of the enire application and at the end
deploying the containerized ML application onto Google Kubernetes Engine (GKE) in GCP and exposing it to internet
and finally running the ML App from different browsers and machines and making predictions
successfully.
This project is based on Heart Failure Prediction dataset from Kaggle.
Here we are predicting the death event or chance of death of a patient
due to heart failure based on 12 clinical features. Those features are
given below:

age, if the patient has anaemia(Decrease of Red Blood cells or Hemoglobin), 
creatinine_phosphokinase (Level of the CPK enzyme in the blood (mcg/L)),
diabetes, ejection_fraction (Percentage of blood leaving the heart at each contraction (percentage)),
high blood pressure, platelets, serum_creatinine (Level of serum creatinine in the blood (mg/dL)),
serum_sodium (Level of serum sodium in the blood (mEq/L)), sex (male or female), smoking, 
time. 

This is a classification problem. Since the dataset was imbalanced, we took care of that
using data resampling technique. In order to take care of the imbalanced training data, we used resampling 
technique (hybrid resampling -> combination Over-Sampling (using SMOTE) and Under-Sampling (using RandomUnderSampler)). 
This improved results quite a lot. Random Forest algorithm gave the best results. 

Results:

Accuracy for the Training Set is 0.9481481481481482

Classification report for the Training Set:

              precision    recall  f1-score   support

         0.0       0.95      0.96      0.95       150
         1.0       0.95      0.93      0.94       120

    accuracy                           0.95       270
   macro avg       0.95      0.95      0.95       270
weighted avg       0.95      0.95      0.95       270

roc_auc_score for the Training Set: 0.9466666666666667
--------------------------------------------------------------

Accuracy for the Test Set: 0.8

Classification report for the Test Set:

              precision    recall  f1-score   support

         0.0       0.80      0.89      0.84        53
         1.0       0.81      0.68      0.74        37

    accuracy                           0.80        90
   macro avg       0.80      0.78      0.79        90
weighted avg       0.80      0.80      0.80        90

roc_auc_score for the Test Set: 0.7812340642529322


For this project, at first we are doing all the necessary data preprocessing steps, model building, training, evaluation, predictions etc. 
in research environment (Jupyter notebook) interactively. Later we wrote production level code in python for the machine learning pipeline. 
We used Scikit Learn Pipeline here for production code. We made sure that the predictions obtained in the Research Environment matched with 
the predictions that we got by wriring and running production level code of the machine learning pipeline. 

Next, we created a web application (with front end UI) using flask and flasgger(Swagger) for serving the Machine Learning model 
(it created API endpoints for predictions where it uses POST method to upload a csv file with 
single patient's feature data, send request to the server and then it displays the model prediction after getting response from the server.
We first successfully tested the web Application for our ML prediction locally through browsers. 
It could predict successfully for the single patient and also multiple patients data.
After that we created a docker image of this Machine Learning pipeline application. We ran and tested 
this docker image from container. Our ML web App worked fine through browser locally
when run from docker container. It could predict successfully for the single patient and also multiple patients data.

Later we created a docker image of the machine learning application, 
uploaded it onto Google Container Registry, created clusters and then deployed this containerized pre-trained machine learning pipeline and Swagger App 
onto Google Kubernetes Engine (GKE) in GCP and then we exposed the application to the Internet.
Then we checked the status of service. Finally we tested our ML Web App for
the prediction of death due to heart failure through different browsers and different machines successfully
with different patients data (with both single patient data and multiple patients data).

 


 









