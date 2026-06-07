import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/traffic_counts.csv")

plt.figure(figsize=(10,5))

plt.plot(
    df["frame"],
    df["total_vehicles"]
)

plt.title("Vehicle Count Over Time")
plt.xlabel("Frame")
plt.ylabel("Total Vehicles")

plt.savefig(
    "outputs/traffic_trend.png"
)

plt.show()