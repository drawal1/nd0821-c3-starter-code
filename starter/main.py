"""
Orchestrator for creating, deleting and checking pipelines (topics, tables and datasets)
Author: Dhar Rawal
"""

import joblib  # type: ignore
import pandas as pd

from fastapi import (  # type: ignore
    Body,
    FastAPI,
    # HTTPException,
    # Response,
    # status,
)
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

from starter.starter.ml.data import process_data
from starter.starter.ml import model

MODEL_FOLDER_PATH = "./starter/model"
LR_MODEL_PTH = f"{MODEL_FOLDER_PATH}/model.pkl"
LR_ENCODER_PTH = f"{MODEL_FOLDER_PATH}/encoder.pkl"
LR_LBL_BNRZR_PTH = f"{MODEL_FOLDER_PATH}/lbl_binarizer.pkl"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

app = FastAPI()


class FeatureData(BaseModel):  # pylint: disable=too-few-public-methods
    """feature data for a person"""

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        """necessary to allow assigning values by field names instead of aliases"""

        allow_population_by_field_name = True


@app.post("/classify_salary/")
def model_inference(  # pylint: disable=too-many-arguments
    feature_data: FeatureData = Body(...),  # ... means required
):
    """
    * Provide the feature data of the person as the post body
    * Based on https://archive.ics.uci.edu/ml/datasets/census+income)
    * Returns the salary classification as response text ("<=50k" or ">50k")
    """

    # get the model, encoder and lb
    lrc_model = joblib.load(LR_MODEL_PTH)
    encoder = joblib.load(LR_ENCODER_PTH)
    lb = joblib.load(LR_LBL_BNRZR_PTH)

    # create dataframe
    data = pd.DataFrame([feature_data.dict(by_alias=True)])

    # we don't have a label. Pass None
    x_inf, _, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_pred = model.inference(lrc_model, x_inf)
    salary_array = lb.inverse_transform(y_pred)

    # return the first value in the salary array,stripping the surrounding single quotes
    return salary_array[0]


@app.get("/", response_class=HTMLResponse)
def home():
    """root path"""
    return """
    <html>
        <head>
            <title>Census Income Classifier</title>
        </head>
        <body>
            <h1>Welcome!</h1>
            <p>This is a logistic regression model which aims to classify whether a
            person
            has an income greater
            than or equal to $50,000 based on various demographc features. The model is
            trained on data from
            the UCI census income data set</p>
            <p>For usage instructions, see the <a href='./docs'>Swagger API</a></p>
        </body>
    </html>
    """
