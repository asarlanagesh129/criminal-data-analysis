import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


train_file = "data/processed/area_train_sample.csv"

model_file = "models/area_random_forest.pkl"
importance_file = "data/processed/area_feature_importance.csv"
chart_file = "data/processed/area_feature_importance.png"


print("Loading training data...")

df = pd.read_csv(train_file)


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
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=2
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


print("\n========== TRAINING ==========")

pipeline.fit(X, y)

print("Training completed.")


# Get transformed feature names
feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()


# Get feature importance
importances = pipeline.named_steps[
    "model"
].feature_importances_


importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})


# Sort by importance
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


print("\n========== TOP 20 FEATURES ==========")

print(
    importance_df.head(20).to_string(index=False)
)


# Save feature importance
importance_df.to_csv(
    importance_file,
    index=False
)


# Plot top 20
top_features = importance_df.head(20).sort_values(
    by="importance"
)


plt.figure(figsize=(12, 8))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title(
    "Top 20 Feature Importances - Crime Area Prediction"
)

plt.xlabel("Feature Importance")

plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    chart_file,
    dpi=300
)

plt.show()


# Save trained pipeline
import joblib

joblib.dump(
    pipeline,
    model_file
)


print("\n========== COMPLETE ==========")

print("Feature importance file:")
print(importance_file)

print("\nFeature importance chart:")
print(chart_file)

print("\nSaved model:")
print(model_file)