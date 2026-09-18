import pandas as pd

input_file = "data/processed/chicago_crime_2018_2025_cleaned.csv"
output_file = "data/processed/chicago_crime_2018_2025_features.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(input_file)

df["date"] = pd.to_datetime(df["date"])

# Time-based features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["hour"] = df["date"].dt.hour

# Day of week number
df["day_of_week_num"] = df["date"].dt.dayofweek

# Weekend flag
df["is_weekend"] = df["day_of_week_num"].apply(
    lambda x: 1 if x >= 5 else 0
)

# Time period
def get_time_period(hour):
    if 0 <= hour <= 5:
        return "Night"
    elif 6 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 16:
        return "Afternoon"
    elif 17 <= hour <= 20:
        return "Evening"
    else:
        return "Late Night"


df["time_period"] = df["hour"].apply(get_time_period)

print("\n========== NEW FEATURES ==========")

print(
    df[
        [
            "date",
            "year",
            "month",
            "day",
            "hour",
            "day_of_week",
            "day_of_week_num",
            "is_weekend",
            "time_period"
        ]
    ].head(10)
)

print("\n========== TIME PERIOD COUNTS ==========")
print(df["time_period"].value_counts())

print("\n========== WEEKEND COUNTS ==========")
print(df["is_weekend"].value_counts())

df.to_csv(output_file, index=False)

print("\n========== FEATURE ENGINEERING COMPLETE ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Output file:", output_file)