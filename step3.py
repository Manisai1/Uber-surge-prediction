import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv("surge_prediction_data.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Feature Engineering
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['weather_encoded'] = df['weather_condition'].astype('category').cat.codes
df['traffic_encoded'] = df['traffic_level'].astype('category').cat.codes

# Define input features and target
features = [
    'ride_distance_km', 'base_fare', 'demand_index',
    'hour', 'dayofweek', 'weather_encoded', 'traffic_encoded'
]
X = df[features]
y = df['is_surge']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Build Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_bal, y_train_bal)

# Evaluate model
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Save model for use in dashboard
joblib.dump(model, "surge_model.pkl")
print("✅ Model saved as surge_model.pkl")