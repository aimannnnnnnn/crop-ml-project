import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# Features
X = data[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

# Target
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/crop_model.pkl")

print("Model saved successfully!")