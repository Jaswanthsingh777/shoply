
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

DATA_PATH = Path(__file__).parent / "data" / "train.csv"

# 1. Load data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# 2. Create date features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek

# 3. Select features and target
features = [
    "store_nbr",
    "family",
    "onpromotion",
    "year",
    "month",
    "day",
    "dayofweek"
]

X = df[features]
y = df["sales"]

# 4. Define feature types
numeric_features = [
    "onpromotion",
    "year",
    "month",
    "day",
    "dayofweek"
]

categorical_features = [
    "store_nbr",
    "family"
]

# 5. Numerical pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Categorical pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

# 7. Combine both pipelines
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

if __name__ == "__main__":
    # Small sample to verify the pipeline
    X_sample = X.head(1000)

    transformed = preprocessor.fit_transform(X_sample)

    print("Preprocessing successful!")
    print("Original shape:", X_sample.shape)
    print("Transformed shape:", transformed.shape)
    print("Target sample:", y.head().tolist())
