import requests
import pandas as pd
from io import StringIO

url = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"

all_data = []

for year in range(2018, 2026):

    print("\nDownloading year:", year)

    params = {
        "$limit": 1000000,
        "$where": f"year = {year}"
    }

    response = requests.get(url, params=params)

    print("Status code:", response.status_code)

    if response.status_code == 200:

        df = pd.read_csv(StringIO(response.text))

        print("Rows downloaded:", len(df))

        all_data.append(df)

    else:

        print("Download failed for:", year)
        print(response.text)

if all_data:

    final_df = pd.concat(all_data, ignore_index=True)

    print("\n========== FINAL DATASET ==========")
    print("Total rows:", len(final_df))
    print("Total columns:", len(final_df.columns))

    print("\nRows by year:")
    print(final_df["year"].value_counts().sort_index())

    output_file = "data/raw/chicago_crime_2018_2025_full.csv"

    final_df.to_csv(
        output_file,
        index=False
    )

    print("\nFile saved successfully:")
    print(output_file)