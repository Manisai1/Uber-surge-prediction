import pandas as pd

# Load dataset
df = pd.read_csv("../surge_prediction_data.csv")

# Remove duplicates
df = df.drop_duplicates()

# Convert timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create hour and day columns
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday

# Print to check
print(df[['timestamp', 'hour', 'dayofweek']].head())

# Encode categorical columns (weather, traffic)
df['weather_encoded'] = df['weather_condition'].astype('category').cat.codes
df['traffic_encoded'] = df['traffic_level'].astype('category').cat.codes

# Optional: View encoded values
print("\nEncoded weather & traffic:")
print(df[['weather_condition', 'weather_encoded', 'traffic_level', 'traffic_encoded']].head())

# Class balance (surge vs non-surge)
print("\nSurge Balance:")
print(df['is_surge'].value_counts())