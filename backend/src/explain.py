import json

import joblib
import numpy as np
import pandas as pd
import shap

from src import config
from src.data_prep import build_features

# Loaded once when the API starts
_model = joblib.load(config.MODEL_PATH)
_columns = joblib.load(config.COLUMNS_PATH)
_explainer = shap.TreeExplainer(_model)


def claim_to_features(claim: dict) -> pd.DataFrame:
    """One claim (a dict of raw fields) -> the exact columns the model was trained on."""
    X = build_features(pd.DataFrame([claim]))
    # A category the model never saw would otherwise become all-zeros silently
    unknown = [c for c in X.columns if c not in _columns]
    if unknown:
        raise ValueError(f"Unrecognised value(s): {unknown}")
    return X.reindex(columns=_columns, fill_value=0)


def predict_and_explain(claim: dict, top_n: int = 8) -> dict:
    X = claim_to_features(claim)
    proba = float(_model.predict_proba(X)[0, 1])

    contrib = _explainer.shap_values(X)[0]              # log-odds push per feature
    base = float(np.ravel(_explainer.expected_value)[0])  # the model's average score

    order = np.argsort(-np.abs(contrib))
    top = [
        {"feature": _columns[i],
         "value": float(X.iloc[0, i]),
         "shap": round(float(contrib[i]), 4)}
        for i in order[:top_n]
    ]
    return {
        "fraud_probability": round(proba, 4),
        "flagged": proba >= config.THRESHOLD,
        "threshold": config.THRESHOLD,
        "base_value": round(base, 4),
        "top_features": top,
        "other_features_shap": round(float(contrib[order[top_n:]].sum()), 4),
    }


if __name__ == "__main__":
    from src.data_prep import load_raw_data

    claim = load_raw_data().iloc[0].to_dict()   # first claim in the dataset
    print(json.dumps(predict_and_explain(claim), indent=2))