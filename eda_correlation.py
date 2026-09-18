import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/processed/chicago_crime_2018_2025_cleaned.csv"

df = pd.read_csv(file_path)

# Select numerical columns
numeric_columns = [
    "year",
    "month",
    "day",
    "hour",
    "beat",
    "district",
    "ward",
    "community_area",
    "latitude",
    "longitude"
]

correlation = df[numeric_columns].corr()

print("========== CORRELATION MATRIX ==========")
print(correlation.round(2))

plt.figure(figsize=(12, 9))

plt.imshow(correlation, cmap="coolwarm", interpolation="nearest")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.title("Correlation Matrix of Numerical Features")

plt.tight_layout()

plt.savefig(
    "data/processed/correlation_matrix.png"
)

plt.show()