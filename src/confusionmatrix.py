import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

preds = model.predict(X_test)

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6,5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(
    range(len(model.classes_)),
    model.classes_
)

plt.yticks(
    range(len(model.classes_)),
    model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.show()