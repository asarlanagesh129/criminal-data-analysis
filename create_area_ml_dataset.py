import pandas as pd

input_file = "data/processed/chicago_crime_2018_2025_features_v2.csv"
output_file = "data/processed/area_ml_dataset.csv"

print("Loading feature dataset...")

df = pd.read_csv(input_file)

print("Original rows:", len(df))


features = [
    "year",
    "month",
    "day",
    "hour",
    "day_of_week_num",
    "is_weekend",
    "district",
    "beat",
    "ward",
    "latitude",
    "longitude",
    "location_description",
    "time_period",
    "season"
]

target = "community_area"


# Keep only rows where the target is available
df = df.dropna(subset=[target])


ml_df = df[features + [target]].copy()


print("\n========== AREA ML DATASET ==========")

print("Rows:", len(ml_df))
print("Columns:", len(ml_df.columns))

print("\n========== FEATURES ==========")
print(features)

print("\n========== TARGET ==========")
print(target)

print("\n========== TARGET DISTRIBUTION ==========")
print(ml_df[target].value_counts().sort_index())

print("\n========== MISSING VALUES ==========")
print(ml_df.isna().sum())


ml_df.to_csv(output_file, index=False)


print("\n========== DATASET CREATED ==========")
print("Output file:", output_file)