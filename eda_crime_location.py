import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

# Get top 10 crime types
top_crimes = df["primary_type"].value_counts().head(10).index

# Filter dataset
filtered_df = df[df["primary_type"].isin(top_crimes)]

# Create crime type vs location table
crime_location = pd.crosstab(
    filtered_df["primary_type"],
    filtered_df["location_description"]
)

# Get the 10 most common locations
top_locations = (
    filtered_df["location_description"]
    .value_counts()
    .head(10)
    .index
)

crime_location = crime_location[top_locations]

print("========== CRIME TYPE VS LOCATION ==========")
print(crime_location)

# Plot
crime_location.plot(
    kind="bar",
    figsize=(15, 8)
)

plt.title("Top Crime Types by Location (2018–2025)")
plt.xlabel("Crime Type")
plt.ylabel("Number of Reported Crimes")
plt.xticks(rotation=45)
plt.legend(
    title="Location",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "data/processed/crime_type_vs_location.png"
)

plt.show()