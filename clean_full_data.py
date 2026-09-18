import pandas as pd

input_file = "data/raw/chicago_crime_2018_2025_full.csv"
output_file = "data/processed/chicago_crime_2018_2025_cleaned.csv"

print("Loading full dataset...")

df = pd.read_csv(input_file)

print("Original rows:", len(df))

# ------------------------------------------------
# 1. Remove completely duplicated rows
# ------------------------------------------------

duplicate_count = df.duplicated().sum()

print("Completely duplicated rows:", duplicate_count)

df = df.drop_duplicates()

# ------------------------------------------------
# 2. Convert date column
# ------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# ------------------------------------------------
# 3. Handle missing location descriptions
# ------------------------------------------------

df["location_description"] = df[
    "location_description"
].fillna("UNKNOWN")

# ------------------------------------------------
# 4. Mark invalid geographic coordinates
# ------------------------------------------------

invalid_geo = (
    (df["latitude"].notna()) &
    (df["longitude"].notna()) &
    (
        (df["latitude"] < 41) |
        (df["latitude"] > 43) |
        (df["longitude"] < -89) |
        (df["longitude"] > -87)
    )
)

print("Invalid geographic records:", invalid_geo.sum())

# Preserve the records but remove invalid coordinates
df.loc[invalid_geo, "latitude"] = pd.NA
df.loc[invalid_geo, "longitude"] = pd.NA
df.loc[invalid_geo, "location"] = pd.NA

# ------------------------------------------------
# 5. Create time features
# ------------------------------------------------

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.day_name()

# ------------------------------------------------
# 6. Final missing-value check
# ------------------------------------------------

print("\n========== FINAL MISSING VALUES ==========")

missing_values = df.isna().sum()

print(missing_values[missing_values > 0])

# ------------------------------------------------
# 7. Save cleaned dataset
# ------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

print("\n========== CLEANING COMPLETE ==========")

print("Final rows:", len(df))
print("Final columns:", len(df.columns))

print("\nYear distribution:")

print(
    df["year"]
    .value_counts()
    .sort_index()
)

print("\nCleaned file:")
print(output_file)