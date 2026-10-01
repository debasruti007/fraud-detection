import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src import config, fairness
from src.data_prep import load_raw_data
from src.explain import predict_and_explain

# Text fields on the form -> dropdowns. Allowed values come from the dataset.
CATEGORICAL = [
    "policy_state", "policy_csl", "insured_education_level", "insured_hobbies",
    "insured_relationship", "incident_type", "collision_type", "incident_severity",
    "authorities_contacted", "incident_state", "incident_city", "property_damage",
    "police_report_available", "auto_make",
]
_df = load_raw_data()
OPTIONS = {c: sorted(_df[c].astype(str).unique().tolist()) for c in CATEGORICAL}


class Claim(BaseModel):
    """The 25 claim fields the model uses. Age, sex, tenure and occupation are
    deliberately NOT here (see the notebook fairness checks)."""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": {
            "policy_state": "OH", "policy_csl": "250/500", "policy_deductable": 1000,
            "policy_annual_premium": 1406.91, "umbrella_limit": 0,
            "insured_education_level": "MD", "insured_hobbies": "sleeping",
            "insured_relationship": "husband", "capital-gains": 53300, "capital-loss": 0,
            "incident_type": "Single Vehicle Collision", "collision_type": "Side Collision",
            "incident_severity": "Major Damage", "authorities_contacted": "Police",
            "incident_state": "SC", "incident_city": "Columbus",
            "incident_hour_of_the_day": 5, "number_of_vehicles_involved": 1,
            "property_damage": "YES", "bodily_injuries": 1, "witnesses": 2,
            "police_report_available": "YES", "total_claim_amount": 71610,
            "auto_make": "Saab", "auto_year": 2004,
        }},
    )

    policy_state: str
    policy_csl: str
    policy_deductable: int = Field(ge=0)
    policy_annual_premium: float = Field(ge=0)
    umbrella_limit: int = Field(ge=0)
    insured_education_level: str
    insured_hobbies: str
    insured_relationship: str
    capital_gains: int = Field(alias="capital-gains", ge=0)
    capital_loss: int = Field(alias="capital-loss")   # entered as a positive amount is fine
    incident_type: str
    collision_type: str
    incident_severity: str
    authorities_contacted: str
    incident_state: str
    incident_city: str
    incident_hour_of_the_day: int = Field(ge=0, le=23)
    number_of_vehicles_involved: int = Field(ge=1, le=4)
    property_damage: str
    bodily_injuries: int = Field(ge=0, le=2)
    witnesses: int = Field(ge=0, le=3)
    police_report_available: str
    total_claim_amount: int = Field(ge=0)
    auto_make: str
    auto_year: int = Field(ge=1980, le=2026)


app = FastAPI(title="Fraud Risk + Fairness Audit API")

# Open for development. Restrict to your frontend's address when you deploy.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    return OPTIONS


@app.post("/predict")
def predict(claim: Claim):
    data = claim.model_dump(by_alias=True)
    for col in CATEGORICAL:
        if data[col] not in OPTIONS[col]:
            raise HTTPException(
                status_code=422,
                detail=f"'{data[col]}' is not a valid {col}. Allowed: {OPTIONS[col]}",
            )
    data["capital-loss"] = -abs(data["capital-loss"])   # the data stores losses as negatives
    try:
        return predict_and_explain(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/fairness")
def fairness_report():
    return fairness.build_report()


@app.get("/metrics")
def metrics():
    with open(config.METRICS_PATH) as f:
        return json.load(f)