import requests
import pandas as pd
from io import StringIO

url = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"

params = {
    "$limit": 100000,
    "$where": "year between 2018 and 2025"
}

print("Downloading Chicago Crime dataset...")

response = requests.get(url, params=params)

print("Status code:", response.status_code)

if response.status_code == 200:
    df = pd.read_csv(StringIO(response.text))

    print("Download completed!")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    df.to_csv(
        "data/raw/chicago_crime_2018_2025.csv",
        index=False
    )

    print("File saved successfully.")
else:
    print("Download failed.")
    print(response.text)