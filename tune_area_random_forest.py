import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------------------
# 1. File paths
# -----------------------------------------

train_file = "data/processed/area_train_sample.csv"
test_file = "data/processed/area_test.csv"

# -----------------------------------------
# 2. Load data
# -----------------------------------------

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
# 6. Random Forest configurations
# -----------------------------------------

models = [
    (
        "RF_100_depth15",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2
        )
    ),

    (
        "RF_100_depth20",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2
        )
    )
]

# -----------------------------------------
# 7. Train and evaluate
# -----------------------------------------

results = []

for name, model in models:

    print()
    print("======================================")
    print("Training:", name)
    print("======================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print("Accuracy:", accuracy)
    print("Accuracy (%):", accuracy * 100)
    print("Weighted F1:", f1)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Accuracy_Percentage": accuracy * 100,
        "Weighted_F1": f1
    })

# -----------------------------------------
# 8. Save results
# -----------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "data/processed/area_random_forest_tuning_results.csv",
    index=False
)

print()
print("======================================")
print("TUNING RESULTS")
print("======================================")

print(results_df)

print()
print("Results saved successfully.")