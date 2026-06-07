import pandas as pd
import numpy as np

df = pd.read_csv("data/traffic_counts.csv")

# Fake time values for portfolio project
df["hour"] = np.random.randint(6, 22, size=len(df))

# Monday=1 ... Sunday=7
df["day_of_week"] = np.random.randint(1, 8, size=len(df))

df.to_csv("data/congestion_data.csv", index=False)

print("Prediction dataset created")