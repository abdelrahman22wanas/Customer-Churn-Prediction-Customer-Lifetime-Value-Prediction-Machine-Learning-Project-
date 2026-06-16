import joblib
import pandas as pd
import numpy as np
from app.config import settings


def load_models():
    model_dir = settings.model_dir.resolve()
    required = [
        "classifier.pkl", "regressor.pkl", "label_encoders.pkl",
        "target_encoder.pkl", "feature_columns.pkl",
        "categorical_features.pkl", "numerical_features.pkl"
    ]
    for f in required:
        if not (model_dir / f).exists():
            raise FileNotFoundError(
                f"Missing model artifact: {f}. Run 'python train.py' first."
            )

    return {
        "classifier": joblib.load(model_dir / "classifier.pkl"),
        "regressor": joblib.load(model_dir / "regressor.pkl"),
        "label_encoders": joblib.load(model_dir / "label_encoders.pkl"),
        "target_encoder": joblib.load(model_dir / "target_encoder.pkl"),
        "feature_columns": joblib.load(model_dir / "feature_columns.pkl"),
        "categorical_features": joblib.load(model_dir / "categorical_features.pkl"),
        "numerical_features": joblib.load(model_dir / "numerical_features.pkl"),
    }


def preprocess_input(data: dict, artifacts: dict) -> np.ndarray:
    feature_cols = artifacts["feature_columns"]
    cat_cols = artifacts["categorical_features"]
    label_encoders = artifacts["label_encoders"]

    df = pd.DataFrame([data])[feature_cols]

    for col in cat_cols:
        raw_val = str(df[col].iloc[0]).lower().strip()
        le = label_encoders[col]
        if raw_val not in le.classes_:
            raise ValueError(
                f"Unknown value '{raw_val}' for '{col}'. "
                f"Expected one of: {list(le.classes_)}"
            )
        df[col] = le.transform([raw_val])

    return df.values.astype(np.float64)
