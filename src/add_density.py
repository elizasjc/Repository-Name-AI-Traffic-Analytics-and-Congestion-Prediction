import pandas as pd

df = pd.read_csv("data/traffic_counts.csv")

def traffic_level(count):

    if count < 20:
        return "Low"

    elif count < 40:
        return "Medium"

    return "High"

df["traffic_density"] = df["total_vehicles"].apply(traffic_level)

df.to_csv("data/traffic_counts.csv", index=False)

print("Traffic density added")