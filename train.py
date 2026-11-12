
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import preprocessor, features

# STEP 1: Load Kaggle dataset
DATA_PATH = Path(__file__).parent / "data" / "train.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("Dataset loaded successfully!")

# STEP 2: Create date features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek

# STEP 3: Use recent sales data
df = df[df["date"] >= "2017-01-01"].copy()
df = df.sort_values("date")

# STEP 4: Split chronologically
train_end = df["date"].quantile(0.70)
val_end = df["date"].quantile(0.85)

train = df[df["date"] <= train_end]
val = df[
    (df["date"] > train_end) &
    (df["date"] <= val_end)
]
test = df[df["date"] > val_end]

X_train = train[features]
y_train = train["sales"]

X_val = val[features]
y_val = val["sales"]

X_test = test[features]
y_test = test["sales"]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

# STEP 5: Build ML pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("algorithm", LinearRegression())
])

# STEP 6: Train
print("\nTraining Linear Regression...")
model.fit(X_train, y_train)

print("Training completed!")

# STEP 7: Validate predictions
predictions = model.predict(X_val)
predictions = predictions.clip(min=0)

mae = mean_absolute_error(y_val, predictions)
rmse = mean_squared_error(y_val, predictions) ** 0.5
r2 = r2_score(y_val, predictions)

# WAPE (Weighted Absolute Percentage Error)
denominator = abs(y_val).sum()

wape = (
    abs(y_val - predictions).sum() / denominator * 100
    if denominator > 0 else float("nan")
)

print("\nMODEL VALIDATION RESULTS")
print("------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))
print("WAPE:", round(wape, 2), "%")

# STEP 8: Save model
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "linear_regression.pkl"
)

print("\nModel saved successfully!")

# STEP 9: Save predictions
results = val[["date", "store_nbr", "family", "sales"]].copy()
results["predicted_sales"] = predictions

results.to_csv(
    MODEL_DIR / "validation_predictions.csv",
    index=False
)

print("Validation predictions saved!")
