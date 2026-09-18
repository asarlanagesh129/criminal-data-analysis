import pandas as pd

file_path = "data/raw/chicago_crime_2018_2025.csv"

df = pd.read_csv(file_path)

print("========== MISSING VALUE INVESTIGATION ==========")

print("\n--- Missing Ward Record ---")

missing_ward = df[df["ward"].isna()]

print(missing_ward[
    [
        "id",
        "case_number",
        "date",
        "primary_type",
        "district",
        "community_area",
        "latitude",
        "longitude"
    ]
].to_string(index=False))


print("\n--- Missing Community Area Records ---")

missing_community = df[df["community_area"].isna()]

print(missing_community[
    [
        "id",
        "case_number",
        "date",
        "primary_type",
        "district",
        "ward",
        "latitude",
        "longitude"
    ]
].to_string(index=False))