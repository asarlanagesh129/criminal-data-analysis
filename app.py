from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime


# =========================================================
# CREATE FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("models/area_random_forest_tuned.pkl")

print("======================================")
print("Crime Area Prediction System")
print("Model loaded successfully")
print("======================================")


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# PREDICTION PAGE
# =========================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # -----------------------------------------
    # OPEN PREDICTION FORM
    # -----------------------------------------

    if request.method == "GET":
        return render_template("predict.html")


    # -----------------------------------------
    # PROCESS PREDICTION
    # -----------------------------------------

    try:

        # =================================================
        # GET USER INPUT
        # =================================================

        year = int(request.form["year"])
        month = int(request.form["month"])
        day = int(request.form["day"])
        hour = int(request.form["hour"])

        district = int(request.form["district"])
        beat = int(request.form["beat"])
        ward = int(request.form["ward"])

        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])
        print("Received Latitude:", latitude)
        print("Received Longitude:", longitude)

        location_description = request.form[
            "location_description"
        ].strip()


        # =================================================
        # INPUT VALIDATION
        # =================================================

        if year < 2018 or year > 2025:
            raise ValueError(
                "Year must be between 2018 and 2025."
            )

        if month < 1 or month > 12:
            raise ValueError(
                "Month must be between 1 and 12."
            )

        if hour < 0 or hour > 23:
            raise ValueError(
                "Hour must be between 0 and 23."
            )

        if district < 1 or district > 61:
            raise ValueError(
                "District must be between 1 and 61."
            )

        if beat < 111 or beat > 6100:
            raise ValueError(
                "Beat must be between 111 and 6100."
            )

        if ward < 1 or ward > 50:
            raise ValueError(
                "Ward must be between 1 and 50."
            )


        # =================================================
        # VALIDATE DATE
        # =================================================

        try:

            date_value = datetime(
                year,
                month,
                day
            )

        except ValueError:

            raise ValueError(
                "Invalid date. Please enter a valid day for the selected month."
            )


        # =================================================
        # VALIDATE CHICAGO COORDINATES
        # =================================================

        if latitude < 41 or latitude > 43:

            raise ValueError(
                "Latitude must be within the Chicago-area range."
            )

        if longitude < -89 or longitude > -87:

            raise ValueError(
                "Longitude must be within the Chicago-area range."
            )


        # =================================================
        # VALIDATE LOCATION
        # =================================================

        if location_description == "":

            raise ValueError(
                "Please select a location description."
            )


        # =================================================
        # CALCULATE DAY OF WEEK
        # =================================================

        day_of_week_num = date_value.weekday()


        # =================================================
        # CALCULATE WEEKEND
        # =================================================

        if day_of_week_num >= 5:

            is_weekend = 1

        else:

            is_weekend = 0


        # =================================================
        # CALCULATE TIME PERIOD
        # =================================================

        if 5 <= hour < 12:

            time_period = "Morning"

        elif 12 <= hour < 17:

            time_period = "Afternoon"

        elif 17 <= hour < 21:

            time_period = "Evening"

        elif 21 <= hour <= 23:

            time_period = "Night"

        else:

            time_period = "Late Night"


        # =================================================
        # CALCULATE SEASON
        # =================================================

        if month in [3, 4, 5]:

            season = "Spring"

        elif month in [6, 7, 8]:

            season = "Summer"

        elif month in [9, 10, 11]:

            season = "Autumn"

        else:

            season = "Winter"


        # =================================================
        # CREATE INPUT DATA
        # =================================================

        data = {

            "year": year,

            "month": month,

            "day": day,

            "hour": hour,

            "day_of_week_num": day_of_week_num,

            "is_weekend": is_weekend,

            "district": district,

            "beat": beat,

            "ward": ward,

            "latitude": latitude,

            "longitude": longitude,

            "location_description": location_description,

            "time_period": time_period,

            "season": season
        }


        # =================================================
        # CREATE DATAFRAME
        # =================================================

        input_data = pd.DataFrame([data])


        # =================================================
        # DEBUG INFORMATION
        # =================================================

        print()

        print("======================================")
        print("FLASK INPUT DATA")
        print("======================================")

        print(
            input_data.to_string(index=False)
        )

        print()

        print("======================================")
        print("MODEL PREDICTION")
        print("======================================")


        # =================================================
        # MAKE PREDICTION
        # =================================================

        prediction = model.predict(
            input_data
        )[0]


        print(
            "Predicted Community Area:",
            prediction
        )

        print("======================================")

        print()


        # =================================================
        # RETURN RESULT TO PREDICT PAGE
        # =================================================

        return render_template(

            "predict.html",

            prediction=int(prediction),

            day_name=date_value.strftime("%A"),

            time_period=time_period,

            season=season

        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        print()

        print("======================================")
        print("ERROR")
        print("======================================")

        print(str(e))

        print("======================================")

        print()

        return render_template(

            "predict.html",

            error=str(e)

        )


# =========================================================
# DASHBOARD PAGE
# =========================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# =========================================================
# WORK PAGE
# =========================================================

@app.route("/work")
def work():

    return render_template(
        "work.html"
    )


# =========================================================
# ABOUT PAGE
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# CONTACT PAGE
# =========================================================

@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


# =========================================================
# START FLASK SERVER
# =========================================================

if __name__ == "__main__":

    app.run(debug=False)