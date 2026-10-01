import json

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src import config
from src.data_prep import prepare_dataset


def flag_stats(y_true, proba, threshold):
    """Share of claims flagged, precision, recall at a given threshold."""
    pred = proba >= threshold
    tp = int((pred & (y_true == 1)).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / int((y_true == 1).sum())
    return float(pred.mean()), float(precision), float(recall)


def main():
    raw_train, raw_test, X_train, X_test, y_train, y_test = prepare_dataset()
    yt_train, yt_test = y_train.to_numpy(), y_test.to_numpy()

    # 1. Honest out-of-fold scores for the training rows (each claim is scored
    #    by a model that never saw it). Used for the threshold check and the audit.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(
        XGBClassifier(**config.MODEL_PARAMS), X_train, y_train,
        cv=cv, method="predict_proba",
    )[:, 1]

    # 2. Final model: trained on the training rows, scored once on the test rows
    model = XGBClassifier(**config.MODEL_PARAMS).fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]

    cv_flag, cv_prec, cv_rec = flag_stats(yt_train, oof, config.THRESHOLD)
    te_flag, te_prec, te_rec = flag_stats(yt_test, test_proba, config.THRESHOLD)
    auc = float(roc_auc_score(yt_test, test_proba))
    ap = float(average_precision_score(yt_test, test_proba))

    print(f"CV   @{config.THRESHOLD}: flagged {cv_flag:.0%}, precision {cv_prec:.2f}, recall {cv_rec:.2f}")
    print(f"TEST @{config.THRESHOLD}: flagged {te_flag:.0%}, precision {te_prec:.2f}, recall {te_rec:.2f}")
    print(f"TEST ROC-AUC {auc:.3f} | avg precision {ap:.3f}")

    # 3. Save everything the API and the fairness dashboard need
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(list(X_train.columns), config.COLUMNS_PATH)

    audit = pd.concat([
        raw_train[config.AUDIT_COLS].assign(fraud=yt_train, score=oof, split="cv"),
        raw_test[config.AUDIT_COLS].assign(fraud=yt_test, score=test_proba, split="test"),
    ], ignore_index=True)
    audit.to_csv(config.AUDIT_PATH, index=False)

    metrics = {
        "threshold": config.THRESHOLD,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "cv": {"flagged": cv_flag, "precision": cv_prec, "recall": cv_rec},
        "test": {"flagged": te_flag, "precision": te_prec, "recall": te_rec,
                 "roc_auc": auc, "avg_precision": ap},
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model, feature columns, fairness_audit.csv and metrics.json to", config.MODEL_DIR)


if __name__ == "__main__":
    main()