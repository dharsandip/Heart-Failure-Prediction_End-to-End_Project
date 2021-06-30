
# data
TRAINING_DATA_FILE = "heart_failure_train_dataset.csv"
TEST_DATA_FILE = "heart_failure_test_dataset.csv"
PIPELINE1_NAME = 'data_preprocessing'
PIPELINE2_NAME = 'random_forest_classification'

TARGET = 'DEATH_EVENT'

# input variables 
FEATURES = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
            'ejection_fraction', 'high_blood_pressure', 'platelets',
            'serum_creatinine', 'serum_sodium', 'sex', 'smoking',
            'time']

# variables to log transform
NUMERICALS_LOG_VARS = ["creatinine_phosphokinase", "serum_creatinine"]


