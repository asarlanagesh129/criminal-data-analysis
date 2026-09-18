import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# 1. LOAD DATA
# ==========================================

train_file = "data/processed/area_train_sample.csv"
test_file = "data/processed/area_test.csv"

print("Loading training and testing data...")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

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


# ==========================================
# 3. NUMERICAL AND CATEGORICAL FEATURES
# ==========================================

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


# ==========================================
# 4. PREPROCESSING
# ==========================================

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


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=2,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
}


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

print()
print("==========================================")
print("MODEL COMPARISON")
print("==========================================")


for model_name, model in models.items():

    print()
    print("------------------------------------------")
    print("Training:", model_name)
    print("------------------------------------------")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


# ==========================================
# 7. CREATE RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)

print()
print("==========================================")
print("FINAL MODEL COMPARISON")
print("==========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print()
print("Results saved to:")
print("model_comparison_results.csv")

print()
print("Model comparison completed.")