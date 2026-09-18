import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_counts = df["day_of_week"].value_counts()

day_counts = day_counts.reindex(day_order)

print("========== CRIMES BY DAY OF WEEK ==========")
print(day_counts)

plt.figure(figsize=(10, 6))

day_counts.plot(kind="bar")

plt.title("Reported Crimes by Day of Week (2018–2025)")
plt.xlabel("Day of Week")
plt.ylabel("Number of Reported Crimes")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "data/processed/crimes_by_day_of_week.png"
)

plt.show()