import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

crime_counts = df["primary_type"].value_counts()

print("========== TOP CRIME TYPES ==========")
print(crime_counts.head(15))

top_crimes = crime_counts.head(15)

plt.figure(figsize=(12, 7))

top_crimes.sort_values().plot(kind="barh")

plt.title("Top 15 Crime Types (2018–2025)")
plt.xlabel("Number of Reported Crimes")
plt.ylabel("Crime Type")

plt.tight_layout()

plt.savefig(
    "data/processed/top_15_crime_types.png"
)

plt.show()