from flask import Flask, render_template, request, session, redirect, url_for
import joblib

app = Flask(__name__)
app.secret_key = "cropsense-language-key"

model = joblib.load("models/crop_model.pkl")


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():
    language = session.get("language", "en")

    return render_template(
        "index.html",
        language=language
    )


# ==========================================
# RECOMMENDATION
# ==========================================

@app.route("/recommend", methods=["GET", "POST"])
def recommend():

    prediction = None

    if request.method == "POST":

        nitrogen = float(request.form["nitrogen"])
        phosphorus = float(request.form["phosphorus"])
        potassium = float(request.form["potassium"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        input_data = [[
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ]]

        prediction = model.predict(input_data)[0]

    return render_template(
        "recommend.html",
        prediction=prediction,
        language=session.get("language", "en")
    )


# ==========================================
# INSIGHTS
# ==========================================

@app.route("/insights")
def insights():

    import pandas as pd

    data = pd.read_csv("data/Crop_recommendation.csv")


    # --------------------------------------
    # FEATURE NAMES
    # --------------------------------------

    feature_names = [
        "Nitrogen (N)",
        "Phosphorus (P)",
        "Potassium (K)",
        "Temperature",
        "Humidity",
        "pH",
        "Rainfall"
    ]


    # --------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------

    feature_importance = model.feature_importances_.tolist()


    # --------------------------------------
    # AVERAGE RAINFALL BY CROP
    # --------------------------------------

    rainfall_by_crop = (
        data.groupby("label")["rainfall"]
        .mean()
        .sort_values(ascending=False)
    )


    crop_names = rainfall_by_crop.index.tolist()


    # --------------------------------------
    # HINDI CROP NAMES
    # --------------------------------------

    if session.get("language", "en") == "hi":

        crop_translation = {

            "apple": "सेब",
            "banana": "केला",
            "blackgram": "उड़द",
            "chickpea": "चना",
            "coconut": "नारियल",
            "coffee": "कॉफी",
            "cotton": "कपास",
            "grapes": "अंगूर",
            "jute": "जूट",
            "kidneybeans": "राजमा",
            "lentil": "मसूर",
            "maize": "मक्का",
            "mango": "आम",
            "mothbeans": "मोठ दाल",
            "mungbean": "मूंग",
            "muskmelon": "खरबूजा",
            "orange": "संतरा",
            "papaya": "पपीता",
            "pigeonpeas": "अरहर",
            "pomegranate": "अनार",
            "rice": "चावल",
            "watermelon": "तरबूज"

        }

        crop_names = [
            crop_translation.get(crop, crop)
            for crop in crop_names
        ]


    # --------------------------------------
    # RAINFALL VALUES
    # --------------------------------------

    crop_rainfall = [
        round(value, 2)
        for value in rainfall_by_crop.values
    ]


    # --------------------------------------
    # MODEL ACCURACY
    # --------------------------------------

    accuracy = 99


    # --------------------------------------
    # SEND DATA TO HTML
    # --------------------------------------

    return render_template(
        "insights.html",
        accuracy=accuracy,
        feature_names=feature_names,
        feature_importance=feature_importance,
        crop_names=crop_names,
        crop_rainfall=crop_rainfall,
        language=session.get("language", "en")
    )


# ==========================================
# ASSISTANT
# ==========================================

@app.route("/assistant")
def assistant():

    return render_template(
        "assistant.html",
        language=session.get("language", "en")
    )


# ==========================================
# ABOUT
# ==========================================

@app.route("/about")
def about():

    return render_template(
        "about.html",
        language=session.get("language", "en")
    )


# ==========================================
# LANGUAGE SWITCH
# ==========================================

@app.route("/set-language/<language>")
def set_language(language):

    if language in ["en", "hi"]:
        session["language"] = language

    return redirect(
        request.referrer or url_for("home")
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)