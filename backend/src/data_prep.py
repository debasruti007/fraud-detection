import pandas as pd
from sklearn.model_selection import train_test_split
from src import config


def load_raw_data() -> pd.DataFrame:
    # keep_default_na=False so the text "None" (no authorities contacted)
    # stays a real category instead of becoming a missing value
    df = pd.read_csv(config.DATA_PATH, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df.replace("?", "Unknown")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw claim rows -> model features. Used for training AND for API requests."""
    X = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns]).copy()
    X["umbrella_limit"] = X["umbrella_limit"].clip(lower=0)
    return pd.get_dummies(X, dtype=int)


def prepare_dataset():
    raw = load_raw_data()
    y = raw[config.TARGET_COL].eq("Y").astype(int)

    raw_train, raw_test, y_train, y_test = train_test_split(
        raw, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )
    raw_train = raw_train.reset_index(drop=True)
    raw_test = raw_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train = build_features(raw_train)
    X_test = build_features(raw_test).reindex(columns=X_train.columns, fill_value=0)
    return raw_train, raw_test, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw_train, raw_test, X_train, X_test, y_train, y_test = prepare_dataset()
    print("Train rows:", len(X_train), "| Test rows:", len(X_test))
    print("Feature count:", X_train.shape[1])
    print("Fraud rate (train):", round(y_train.mean(), 3))