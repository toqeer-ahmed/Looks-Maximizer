import pandas as pd

try:
    df = pd.read_parquet(r"ml_pipeline/HuggingFace/1.25/train-00000-of-00004-e715178553977907.parquet")
    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nUnique Age:", df['age'].unique())
    print("Unique Gender:", df['gender'].unique())
    print("Unique Race:", df['race'].unique())
except Exception as e:
    print(f"Error reading parquet: {e}")
