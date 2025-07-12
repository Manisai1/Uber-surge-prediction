import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("surge_prediction_data.csv")

# Preprocessing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Top 10 pickup zones by volume
top_locations = df['pickup_location'].value_counts().head(10)

# Plot 1: Top pickup zones
plt.figure(figsize=(10, 5))
sns.barplot(x=top_locations.index, y=top_locations.values, palette='coolwarm')
plt.title("Top 10 High-Demand Pickup Locations")
plt.ylabel("Ride Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Heatmap of hourly ride demand by location
pivot = df.pivot_table(index='pickup_location', columns='hour', values='ride_id', aggfunc='count')
top10_pivot = pivot.loc[top_locations.index]

plt.figure(figsize=(12, 6))
sns.heatmap(top10_pivot.fillna(0), cmap="YlGnBu", annot=True, fmt=".0f")
plt.title("Hourly Ride Demand by Pickup Location")
plt.xlabel("Hour of Day")
plt.ylabel("Pickup Location")
plt.tight_layout()
plt.show()