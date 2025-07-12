import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("surge_prediction_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# 1. Surge % by hour
surge_by_hour = df.groupby('hour')['is_surge'].mean()

# 2. Surge % by day
surge_by_day = df.groupby('dayofweek')['is_surge'].mean()

# 3. Surge % by weather
surge_by_weather = df.groupby('weather_condition')['is_surge'].mean()

# Plot all 3
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x=surge_by_hour.index, y=surge_by_hour.values, ax=axs[0])
axs[0].set_title("Surge % by Hour of Day")

sns.barplot(x=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], y=surge_by_day.values, ax=axs[1])
axs[1].set_title("Surge % by Day of Week")

sns.barplot(x=surge_by_weather.index, y=surge_by_weather.values, ax=axs[2])
axs[2].set_title("Surge % by Weather")

plt.tight_layout()
plt.show()