import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

district_counts = df["district"].value_counts().sort_index()

print("========== CRIMES BY DISTRICT ==========")
print(district_counts)

top_districts = df["district"].value_counts().head(15)

print("\n========== TOP 15 DISTRICTS ==========")
print(top_districts)

plt.figure(figsize=(12, 7))

top_districts.sort_values().plot(kind="barh")

plt.title("Top 15 Police Districts by Reported Crimes (2018–2025)")
plt.xlabel("Number of Reported Crimes")
plt.ylabel("District")

plt.tight_layout()

plt.savefig(
    "data/processed/top_15_crime_districts.png"
)

plt.show()