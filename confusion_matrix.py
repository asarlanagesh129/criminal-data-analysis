import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib


# File paths
test_file = "data/processed/balanced_test.csv"
model_file = "models/random_forest_balanced.pkl"

output_file = "data/processed/random_forest_confusion_matrix.png"


print("Loading test data...")

test_df = pd.read_csv(test_file)

target = "primary_type"

features = [
    "month",
    "day",
    "hour",
    "day_of_week_num",
    "is_weekend",
    "district",
    "beat",
    "ward",
    "community_area",
    "location_description"
]


X_test = test_df[features]
y_test = test_df[target]


print("Loading Random Forest model...")

model = joblib.load(model_file)


print("\n========== PREDICTION ==========")

y_pred = model.predict(X_test)

print("Prediction completed.")


# Get class names
classes = sorted(y_test.unique())


# Create confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)


print("\n========== CONFUSION MATRIX ==========")
print(cm)


# Plot
plt.figure(figsize=(18, 15))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    xticks_rotation=90,
    values_format="d"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    output_file,
    dpi=150
)

plt.show()

print("\n========== COMPLETE ==========")
print("Confusion matrix saved to:")
print(output_file)