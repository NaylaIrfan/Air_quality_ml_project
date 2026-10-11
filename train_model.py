
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/air_quality.csv")
MODEL_DIR = Path("models")

FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

TARGET = "AQI_Bucket"


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            "Dataset missing. Run python generate_demo_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"CSV must contain the target column {TARGET!r}."
        )

    features = [
        column for column in FEATURES
        if column in df.columns
    ]

    if not features:
        raise ValueError(
            "No supported pollutant columns found."
        )

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET] != ""]
    df = df.dropna(subset=features, how="all")

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "At least two different air-quality labels are required."
        )

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET]

    stratify = (
        y if y.value_counts().min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Total rows: {len(df)}")
    print(f"Features: {', '.join(features)}")
    print(f"Test accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_DIR / "air_quality_model.joblib"
    )

    metadata = {
        "features": features,
        "classes": [str(c) for c in model.classes_],
        "accuracy": round(float(accuracy), 4),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "data_note": (
            "Synthetic demo data. Use real labeled data "
            "for meaningful real-world evaluation."
        )
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nModel saved successfully in the models folder.")


if __name__ == "__main__":
    main()
