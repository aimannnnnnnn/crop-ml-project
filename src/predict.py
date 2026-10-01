import joblib

# Load the trained model
model = joblib.load("models/crop_model.pkl")

print("🌱 Crop Recommendation System")
print("--------------------------------")

# Get input from the user
N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorus (P): "))
K = float(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter pH: "))
rainfall = float(input("Enter Rainfall: "))

# Make prediction
prediction = model.predict([[
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
]])

print("\nRecommended crop:", prediction[0])