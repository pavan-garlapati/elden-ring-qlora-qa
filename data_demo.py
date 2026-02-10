import pandas as pd
import os
import glob

data_dir = "data"  # Change this to your data folder path

csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

if not csv_files:
    print(f"No CSV files found in '{data_dir}/'")
    print("Make sure your Kaggle CSVs are in the data/ folder.")
else:
    for csv_path in sorted(csv_files):
        fname = os.path.basename(csv_path)
        try:
            df = pd.read_csv(csv_path)
            print(f"\n{'='*70}")
            print(f"FILE: {fname}")
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
            print(f"  Columns: {list(df.columns)}")
            print(f"\n  Column dtypes & non-null counts:")
            for col in df.columns:
                non_null = df[col].notna().sum()
                print(f"    {col:30s} | {str(df[col].dtype):10s} | {non_null}/{len(df)} non-null")
            print(f"\n  First row sample:")
            for col in df.columns:
                val = df[col].iloc[0] if len(df) > 0 else "N/A"
                val_str = str(val)[:80]
                print(f"    {col}: {val_str}")
        except Exception as e:
            print(f"\nFILE: {fname} — ERROR: {e}")