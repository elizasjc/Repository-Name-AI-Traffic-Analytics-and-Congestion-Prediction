import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/congestion_data.csv")

X = df[
    [
        "hour",
        "day_of_week",
        "total_vehicles"
    ]
]

y = df["traffic_density"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


print("Model saved successfully")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")

sample = [[8, 2, 55]]

prediction = model.predict(sample)

import matplotlib.pyplot as plt

features = [
    "hour",
    "day_of_week",
    "total_vehicles"
]

importance = model.feature_importances_

plt.figure(figsize=(6,4))
plt.bar(features, importance)

plt.title("Feature Importance")
plt.ylabel("Importance Score")

plt.savefig(
    "outputs/feature_importance.png"
)

plt.show()

joblib.dump(
    model,
    "models/congestion_model.pkl"
)

print(
    "Predicted Traffic Density:",
    prediction[0]
)