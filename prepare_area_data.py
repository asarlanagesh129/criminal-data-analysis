import pandas as pd
from sklearn.model_selection import train_test_split


input_file = "data/processed/area_ml_dataset.csv"

train_file = "data/processed/area_train.csv"
test_file = "data/processed/area_test.csv"


print("Loading area ML dataset...")

df = pd.read_csv(input_file)

print("Total rows:", len(df))


features = [
    "year",
    "month",
    "day",
    "hour",
    "day_of_week_num",
    "is_weekend",
    "district",
    "beat",
    "ward",
    "latitude",
    "longitude",
    "location_description",
    "time_period",
    "season"
]

target = "community_area"


X = df[features]
y = df[target]


print("\n========== DATA ==========")
print("Features:", len(features))
print("Target:", target)
print("Classes:", y.nunique())


# Remove classes that have fewer than 2 records
class_counts = y.value_counts()

valid_classes = class_counts[class_counts >= 2].index

valid_rows = y.isin(valid_classes)

X = X[valid_rows]
y = y[valid_rows]


print("\nRows after rare-class handling:", len(X))
print("Classes:", y.nunique())


# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


train_df = X_train.copy()
train_df[target] = y_train

test_df = X_test.copy()
test_df[target] = y_test


print("\n========== TRAIN / TEST SPLIT ==========")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


print("\n========== TRAINING DISTRIBUTION ==========")
print(y_train.value_counts().sort_index())


print("\n========== TESTING DISTRIBUTION ==========")
print(y_test.value_counts().sort_index())


train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)


print("\n========== COMPLETE ==========")

print("Training file:")
print(train_file)

print("\nTesting file:")
print(test_file)