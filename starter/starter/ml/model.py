import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


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
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
    lrc.fit(X_train, y_train)

    return lrc


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
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

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
    return model.predict(X)

def save(model, encoder, lb, folder_path):
    """save the model, encoder and lb to pkl files in given folder

    Inputs
    ------
    model : ???
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    folder_path: path to folder where the pkl files will be saved
    Returns
    -------
    Nothing
    """
    joblib.dump(model, f"{folder_path}/model.pkl")
    joblib.dump(encoder, f"{folder_path}/encoder.pkl")
    joblib.dump(lb, f"{folder_path}/lbl_binarizer.pkl")

def load(folder_path):
    """load the model, encoder and lb from pkl files in given folder and return them

    Inputs
    ------
    folder_path: path to folder where the pkl files will be saved
    Returns
    -------
    model : ???
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    lrc_model = joblib.load(f"{folder_path}/model.pkl")
    encoder = joblib.load(f"{folder_path}/encoder.pkl")
    lb = joblib.load(f"{folder_path}/lbl_binarizer.pkl")

    return (lrc_model, encoder, lb)
