import pandas as pd
import joblib

# -----------------------------------------
# 1. Load tuned model
# -----------------------------------------

model = joblib.load(
    "models/area_random_forest_tuned.pkl"
)

print("======================================")
print("TUNED AREA MODEL TEST")
print("Model loaded successfully")
print("======================================")


# -----------------------------------------
# 2. Test Case 1
# -----------------------------------------

test_1 = pd.DataFrame([{
    "year": 2018,
    "month": 1,
    "day": 10,
    "hour": 8,
    "day_of_week_num": 2,
    "is_weekend": 0,
    "district": 1,
    "beat": 111,
    "ward": 1,
    "latitude": 41.88,
    "longitude": -87.63,
    "location_description": "STREET",
    "time_period": "Morning",
    "season": "Winter"
}])


# -----------------------------------------
# 3. Test Case 2
# -----------------------------------------

test_2 = pd.DataFrame([{
    "year": 2022,
    "month": 6,
    "day": 15,
    "hour": 14,
    "day_of_week_num": 2,
    "is_weekend": 0,
    "district": 1,
    "beat": 111,
    "ward": 1,
    "latitude": 41.88,
    "longitude": -87.63,
    "location_description": "STREET",
    "time_period": "Afternoon",
    "season": "Summer"
}])


# -----------------------------------------
# 4. Test Case 3
# -----------------------------------------

test_3 = pd.DataFrame([{
    "year": 2025,
    "month": 12,
    "day": 25,
    "hour": 22,
    "day_of_week_num": 3,
    "is_weekend": 0,
    "district": 20,
    "beat": 2021,
    "ward": 40,
    "latitude": 41.99,
    "longitude": -87.70,
    "location_description": "APARTMENT",
    "time_period": "Night",
    "season": "Winter"
}])


# -----------------------------------------
# 5. Make predictions
# -----------------------------------------

prediction_1 = model.predict(test_1)[0]
prediction_2 = model.predict(test_2)[0]
prediction_3 = model.predict(test_3)[0]


# -----------------------------------------
# 6. Display results
# -----------------------------------------

print()
print("Test Case 1")
print("Predicted Community Area:", prediction_1)

print()
print("Test Case 2")
print("Predicted Community Area:", prediction_2)

print()
print("Test Case 3")
print("Predicted Community Area:", prediction_3)

print()
print("======================================")
print("TEST COMPLETED")
print("======================================")