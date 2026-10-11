
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/air_quality_model.joblib")
META_PATH = Path("models/model_metadata.json")

st.set_page_config(
    page_title="Air Quality Classifier",
    page_icon="🌿"
)

st.title("🌿 Air Quality Level Classifier")
st.write("Enter pollutant measurements to predict an air-quality category.")
st.caption("Educational project only. Not an official AQI service.")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.warning("The model has not been trained yet. Run these commands first:")
    st.code(
        "python generate_demo_data.py\n"
        "python train_model.py"
    )
    st.stop()

model = joblib.load(MODEL_PATH)

metadata = json.loads(
    META_PATH.read_text(encoding="utf-8")
)

features = metadata["features"]

defaults = {
    "PM2.5": 35,
    "PM10": 70,
    "NO": 10,
    "NO2": 20,
    "NOx": 30,
    "NH3": 15,
    "CO": 0.8,
    "SO2": 10,
    "O3": 30,
    "Benzene": 1,
    "Toluene": 3,
    "Xylene": 1
}

with st.form("prediction_form"):
    st.subheader("Pollutant Measurements")

    columns = st.columns(2)
    values = {}

    for i, feature in enumerate(features):
        with columns[i % 2]:
            unit = "mg/m³" if feature == "CO" else "µg/m³"

            values[feature] = st.number_input(
                f"{feature} ({unit})",
                min_value=0.0,
                max_value=10000.0,
                value=float(defaults.get(feature, 0)),
                step=1.0
            )

    submitted = st.form_submit_button(
        "Predict Air Quality"
    )

if submitted:
    input_data = pd.DataFrame(
        [values],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Air Quality Category: {prediction}"
    )

    if hasattr(model, "predict_proba"):
        probabilities = pd.DataFrame({
            "Category": model.classes_,
            "Probability": model.predict_proba(input_data)[0]
        }).set_index("Category")

        st.subheader("Model Probability Estimates")
        st.bar_chart(probabilities)

    st.caption(
        "Predictions may be inaccurate and should not be "
        "used for health or emergency decisions."
    )

with st.expander("About This Project"):
    st.write("Algorithm: Random Forest Classifier")
    st.write("Missing values: Median imputation")
    st.write(
        f"Input features: {', '.join(features)}"
    )
    st.write(
        f"Training rows: {metadata.get('training_rows', 'unknown')}"
    )
    st.write(
        f"Test accuracy: {metadata.get('accuracy', 'unknown')}"
    )
    st.write(metadata.get("data_note", ""))
