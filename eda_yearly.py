import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

# Count crimes by year
yearly_crimes = df.groupby("year").size()

print("========== CRIMES BY YEAR ==========")

print(yearly_crimes)

# Create chart
plt.figure(figsize=(10, 6))

plt.plot(
    yearly_crimes.index,
    yearly_crimes.values,
    marker="o"
)

plt.title("Reported Crimes by Year (2018–2025)")
plt.xlabel("Year")
plt.ylabel("Number of Reported Crimes")

plt.xticks(yearly_crimes.index)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "data/processed/crimes_by_year.png"
)

plt.show()