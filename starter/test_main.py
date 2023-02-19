"""
Author: Dhar Rawal
Test code to test FastAPI
"""

from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app, FeatureData

# Instantiate the testing client with our app.
client = TestClient(app)


# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    """Test the root path"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() is not None


def test_model_inference_salary_over50k():
    """Test where salary is over 50k"""
    ft_data = FeatureData(
        age=90,
        workclass="Local-gov",
        fnlgt=227796,
        education="Masters",
        education_num=14,
        marital_status="Married-civ-spouse",
        occupation="Exec-managerial",
        relationship="Husband",
        race="White",
        sex="Male",
        capital_gain="20051",
        capital_loss=0,
        hours_per_week=60,
        native_country="United-States",
    )

    r = client.post("/classify_salary/", json=ft_data.dict(by_alias=True))
    assert r.status_code == 200
    assert ">50K" in r.json()


def test_model_inference_salary_not_over50k():
    """Test where salary is under 50k"""
    ft_data = FeatureData(
        age=90,
        workclass="?",
        fnlgt=166343,
        education="1st-4th",
        education_num=2,
        marital_status="Widowed",
        occupation="?",
        relationship="Not-in-family",
        race="Black",
        sex="Female",
        capital_gain="0",
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States",
    )

    r = client.post("/classify_salary/", json=ft_data.dict(by_alias=True))
    assert "<=50K" in r.json()
