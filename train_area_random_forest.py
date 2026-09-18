import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


train_file = "data/processed/area_train_sample.csv"
test_file = "data/processed/area_test.csv"


print("Loading training and testing data...")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


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


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


numeric_features = [
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
    "longitude"
]

categorical_features = [
    "location_description",
    "time_period",
    "season"
]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


model = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


print("\n========== TRAINING RANDOM FOREST ==========")

pipeline.fit(X_train, y_train)

print("Training completed.")

joblib.dump(pipeline, "models/area_random_forest.pkl")

print("Model saved to: models/area_random_forest.pkl")

print("\n========== PREDICTION ==========")

y_pred = pipeline.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)


print("\n========== RESULTS ==========")

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)