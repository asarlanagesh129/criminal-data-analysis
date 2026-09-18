import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------------
# 1. File paths
# -----------------------------------------

train_file = "data/processed/area_train.csv"
test_file = "data/processed/area_test.csv"

model_file = "models/area_random_forest_tuned.pkl"


# -----------------------------------------
# 2. Load data
# -----------------------------------------

print("Loading training data...")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# -----------------------------------------
# 3. Features and target
# -----------------------------------------

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


# -----------------------------------------
# 4. Feature types
# -----------------------------------------

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


# -----------------------------------------
# 5. Preprocessing
# -----------------------------------------

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


# -----------------------------------------
# 6. Tuned Random Forest
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=2
)


# -----------------------------------------
# 7. Create pipeline
# -----------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------------------
# 8. Train model
# -----------------------------------------

print()
print("======================================")
print("TRAINING TUNED RANDOM FOREST")
print("======================================")

print("Trees: 100")
print("Maximum depth: 20")
print("Minimum samples per leaf: 2")

pipeline.fit(X_train, y_train)


# -----------------------------------------
# 9. Make predictions
# -----------------------------------------

print()
print("Making predictions...")

y_pred = pipeline.predict(X_test)


# -----------------------------------------
# 10. Evaluate model
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print()
print("======================================")
print("MODEL RESULTS")
print("======================================")

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -----------------------------------------
# 11. Save tuned model
# -----------------------------------------

joblib.dump(
    pipeline,
    model_file
)

print()
print("======================================")
print("MODEL SAVED")
print("======================================")

print("Saved to:", model_file)
print("======================================")