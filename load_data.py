
from pathlib import Path
import pandas as pd

# Locate the Kaggle dataset
DATA_PATH = Path(__file__).parent / "data" / "train.csv"

# Check whether the file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

# Load the dataset
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Display basic information
print("\nDATASET LOADED SUCCESSFULLY!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDate range:")
print("Start:", df["date"].min())
print("End:", df["date"].max())

print("\nSales statistics:")
print(df["sales"].describe())

print("\nUnique stores:")
print(df["store_nbr"].nunique())

print("\nUnique product families:")
print(df["family"].nunique())
