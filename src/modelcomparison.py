import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    results.append([name, accuracy])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy"]
)

print(results_df)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.bar(results_df["Model"], results_df["Accuracy"])

plt.title("Model Comparison")
plt.ylabel("Accuracy")

plt.savefig("outputs/model_comparison.png")
plt.show()