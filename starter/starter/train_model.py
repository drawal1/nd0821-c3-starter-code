# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import train_model

MODEL_FOLDER_PATH = './starter/model'
LR_MODEL_PTH = f'{MODEL_FOLDER_PATH}/model.pkl'
LR_ENCODER_PTH = f'{MODEL_FOLDER_PATH}/encoder.pkl'
LR_LBL_BNRZR_PTH = f'{MODEL_FOLDER_PATH}/lbl_binarizer.pkl'

# Add code to load in the data.
data = pd.read_csv('./starter/data/census.csv')

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, \
        encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

joblib.dump(model, LR_MODEL_PTH)
joblib.dump(encoder, LR_ENCODER_PTH)
joblib.dump(lb, LR_LBL_BNRZR_PTH)
