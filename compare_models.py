
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import preprocessor, features
from models import get_models

BASE = Path(__file__).parent
def convert_to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

# 1. Load dataset
df = pd.read_csv(
    BASE / "data" / "train.csv",
    parse_dates=["date"]
)

# 2. Feature engineering
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek

# 3. Use recent data
end_date = df["date"].max()
start_date = end_date - pd.Timedelta(days=90)

df = df[df["date"] >= start_date].copy()
df = df.sort_values("date")

# 4. Chronological train/validation split
dates = np.sort(df["date"].unique())
split_index = int(len(dates) * 0.8)
cutoff = dates[split_index]

train = df[df["date"] < cutoff]
val = df[df["date"] >= cutoff]

# Use one fixed training sample for all models
if len(train) > 50000:
    train = train.sample(
        n=50000, random_state=42
    ).sort_values("date")

X_train = train[features]
y_train = train["sales"]

X_val = val[features]
y_val = val["sales"]

print("Training rows:", len(train))
print("Validation rows:", len(val))

# 5. Train all 8 models
results = []
best_score = float("inf")
best_name = None

output_dir = BASE / "results"
output_dir.mkdir(exist_ok=True)

model_dir = BASE / "models"
model_dir.mkdir(exist_ok=True)

for name, algorithm in get_models().items():
    print(f"\nTraining: {name}")
    start = time.time()

    
    steps = [
        ("preprocessing", clone(preprocessor))
    ]

    if name == "Hist Gradient Boosting":
        steps.append((
            "to_dense",
            FunctionTransformer(convert_to_dense)
        ))

    steps.append(("algorithm", algorithm))

    pipeline = Pipeline(steps)


    try:
        pipeline.fit(X_train, y_train)

        predictions = np.maximum(
            pipeline.predict(X_val), 0
        )

        mae = mean_absolute_error(
            y_val, predictions
        )

        rmse = np.sqrt(
            mean_squared_error(y_val, predictions)
        )

        r2 = r2_score(y_val, predictions)

        denominator = np.abs(y_val).sum()
        wape = (
            np.abs(y_val.to_numpy() - predictions).sum()
            / denominator * 100
            if denominator > 0 else np.nan
        )

        elapsed = time.time() - start

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "WAPE (%)": round(wape, 2),
            "Time (s)": round(elapsed, 2)
        })

        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.4f}")
        print(f"WAPE: {wape:.2f}%")

        if mae < best_score:
            best_score = mae
            best_name = name

            joblib.dump(
                pipeline,
                model_dir / "best_model.pkl"
            )

    except Exception as error:
        print(f"FAILED: {name}: {error}")
        results.append({
            "Model": name,
            "Error": str(error)
        })

# 6. Save comparison
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "MAE", na_position="last"
)

comparison.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nBest model:", best_name)
print("Results saved to results/model_comparison.csv")
print("Best pipeline saved to models/best_model.pkl")
