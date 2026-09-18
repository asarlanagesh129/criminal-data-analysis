import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

# Remove missing community areas
hotspot_data = df.dropna(subset=["community_area"])

# Count crimes by community area
community_counts = (
    hotspot_data["community_area"]
    .value_counts()
    .sort_values(ascending=False)
)

print("========== TOP CRIME HOTSPOT AREAS ==========")
print(community_counts.head(15))

# Plot top 15 community areas
top_areas = community_counts.head(15)

plt.figure(figsize=(12, 7))

top_areas.sort_values().plot(kind="barh")

plt.title("Top 15 Community Areas by Reported Crimes (2018–2025)")
plt.xlabel("Number of Reported Crimes")
plt.ylabel("Community Area")

plt.tight_layout()

plt.savefig(
    "data/processed/top_15_crime_hotspots.png"
)

plt.show()