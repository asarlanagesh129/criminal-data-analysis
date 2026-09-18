import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

hour_counts = df["hour"].value_counts().sort_index()

print("========== CRIMES BY HOUR ==========")
print(hour_counts)

plt.figure(figsize=(12, 6))

plt.plot(
    hour_counts.index,
    hour_counts.values,
    marker="o"
)

plt.title("Reported Crimes by Hour of Day (2018–2025)")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Reported Crimes")

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "data/processed/crimes_by_hour.png"
)

plt.show()