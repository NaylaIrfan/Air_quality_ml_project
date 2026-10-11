
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 800

# Generate synthetic pollutant measurements.
pm25 = rng.gamma(2.2, 35, n).clip(2, 350)

pm10 = (
    pm25 * rng.uniform(1.1, 2.1, n)
    + rng.normal(0, 12, n)
).clip(5, 500)

no = rng.gamma(1.5, 8, n).clip(0.1, 100)
no2 = rng.gamma(2, 12, n).clip(1, 150)

nox = (
    no + no2 + rng.normal(2, 4, n)
).clip(1, 200)

nh3 = rng.gamma(2, 8, n).clip(1, 100)
co = rng.gamma(1.5, 0.6, n).clip(0.1, 10)
so2 = rng.gamma(2, 6, n).clip(1, 100)
o3 = rng.gamma(2, 15, n).clip(1, 160)

benzene = rng.gamma(1.5, 2, n).clip(0.1, 30)
toluene = rng.gamma(1.8, 4, n).clip(0.1, 60)
xylene = rng.gamma(1.4, 1.5, n).clip(0.1, 25)

# Create demo labels using a simple artificial rule.
score = np.maximum.reduce([
    pm25 / 2.5,
    pm10 / 4,
    no2 / 1.5,
    o3 / 1.5,
    so2 / 1.5
])

labels = np.select(
    [
        score < 25,
        score < 50,
        score < 100,
        score < 150,
        score < 200
    ],
    [
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "Very Poor"
    ],
    default="Severe"
)

df = pd.DataFrame({
    "PM2.5": pm25,
    "PM10": pm10,
    "NO": no,
    "NO2": no2,
    "NOx": nox,
    "NH3": nh3,
    "CO": co,
    "SO2": so2,
    "O3": o3,
    "Benzene": benzene,
    "Toluene": toluene,
    "Xylene": xylene,
    "AQI_Bucket": labels
})

# Add missing values to demonstrate data preprocessing.
for column in ["PM2.5", "NO2", "CO"]:
    indexes = rng.choice(
        n,
        size=12,
        replace=False
    )
    df.loc[indexes, column] = np.nan

output_path = Path("data/air_quality.csv")
output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(output_path, index=False)

print(f"Dataset created: {output_path}")
print(f"Total records: {len(df)}")
print(
    "Note: This is synthetic demo data, "
    "not real air-quality data."
)
