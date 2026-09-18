import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

location_counts = df["location_description"].value_counts()

print("========== TOP 15 CRIME LOCATIONS ==========")
print(location_counts.head(15))

top_locations = location_counts.head(15)

plt.figure(figsize=(12, 7))

top_locations.sort_values().plot(kind="barh")

plt.title("Top 15 Crime Locations (2018–2025)")
plt.xlabel("Number of Reported Crimes")
plt.ylabel("Location")

plt.tight_layout()

plt.savefig(
    "data/processed/top_15_crime_locations.png"
)

plt.show()