# Air_quality_ml_project
A beginner-friendly Machine Learning project that classifies air quality levels using Python, Pandas, and the Random Forest algorithm, with an interactive web application for air quality prediction.
# Air Quality Level Classification Using Machine Learning

## About the Project
This is a beginner-friendly Machine Learning project that predicts air quality categories using pollutant measurements. It uses the Random Forest classification algorithm and Python.

## Features
- Predicts air quality categories.
- Handles missing data.
- Trains a Random Forest model.
- Evaluates model performance.
- Provides a simple web interface using Streamlit.
- Displays prediction probabilities.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Project Structure
```text
air-quality-ml-project/
├── app.py
├── train_model.py
├── generate_demo_data.py
├── requirements.txt
├── README.md
├── data/
│   └── air_quality.csv
└── models/
```

## Installation and Setup

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate the demo dataset
```bash
python generate_demo_data.py
```

### Step 3: Train the model
```bash
python train_model.py
```

### Step 4: Run the application
```bash
streamlit run app.py
```

## How It Works
1. Loads air pollution data from a CSV file.
2. Selects pollutant measurements as input features.
3. Handles missing values using median imputation.
4. Trains a Random Forest classification model.
5. Evaluates the model using test data.
6. Predicts an air quality category from new measurements.

## Dataset
The included dataset is synthetically generated for demonstration and software testing. It does not represent real-world pollution measurements. For meaningful evaluation, replace it with a real labeled air-quality dataset and mention the source.

## Limitations
- Prediction quality depends on the training dataset.
- Synthetic data cannot establish real-world accuracy.
- This project is educational and is not an official air-quality monitoring or health advisory system.

## Future Improvements
- Use real-world air-quality data.
- Compare different Machine Learning algorithms.
- Add data visualizations.
- Deploy the application online.

## Author
**Name:** Nayla Irfan  
**Course:** B.Tech CSE, First Semester  
**University:** COER University
