import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])

monthly_crimes = df.groupby(df["date"].dt.month).size()

month_names = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

monthly_crimes.index = [month_names[i - 1] for i in monthly_crimes.index]

print("========== CRIMES BY MONTH ==========")
print(monthly_crimes)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_crimes.index,
    monthly_crimes.values,
    marker="o"
)

plt.title("Reported Crimes by Month (2018–2025)")
plt.xlabel("Month")
plt.ylabel("Number of Reported Crimes")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "data/processed/crimes_by_month.png"
)

plt.show()