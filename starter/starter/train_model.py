# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, save

MODEL_FOLDER_PATH = "./starter/model"

# Add code to load in the data.
data = pd.read_csv("./starter/data/census.csv")

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
# get the encoder and lb
_, _, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, _ = train_test_split(data, test_size=0.20, stratify=data["salary"])

# Process the training data with the encoder and lb
X_train, y_train, _, _ = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)

save(model, encoder, lb, MODEL_FOLDER_PATH)
