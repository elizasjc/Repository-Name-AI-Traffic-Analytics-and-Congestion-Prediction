import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/traffic_counts.csv")

vehicle_totals = {
    "Cars": df["cars"].sum(),
    "Motorcycles": df["motorcycles"].sum(),
    "Buses": df["buses"].sum(),
    "Trucks": df["trucks"].sum()
}

print(vehicle_totals)

plt.figure(figsize=(8,5))
plt.bar(vehicle_totals.keys(), vehicle_totals.values())

plt.title("Vehicle Type Distribution")
plt.xlabel("Vehicle Type")
plt.ylabel("Total Count")

plt.savefig("outputs/vehicle_distribution.png")
plt.show()