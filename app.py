from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained ML model
model = joblib.load("models/crop_model.pkl")
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the HTML form
    nitrogen = float(request.form["nitrogen"])
    phosphorus = float(request.form["phosphorus"])
    potassium = float(request.form["potassium"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # Arrange the values in the same order used while training
    input_data = [[
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall
    ]]

    # Ask the ML model for a prediction
    prediction = model.predict(input_data)

    # Get the predicted crop
    crop = prediction[0]

    return render_template(
        "index.html",
        prediction=crop
    )


if __name__ == "__main__":
    app.run(debug=True)
