import pandas as pd
import joblib
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("surge_model.pkl")
df = pd.read_csv("surge_prediction_data.csv")

# Feature engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['weather_encoded'] = df['weather_condition'].astype('category').cat.codes
df['traffic_encoded'] = df['traffic_level'].astype('category').cat.codes

# Streamlit setup
st.set_page_config(page_title="Uber Surge & Demand Dashboard", layout="wide")
st.title(" Uber Surge Price & Driver Demand Prediction")

# Tabs
tab1, tab2 = st.tabs([" Predict Surge (Customer)", "🗺 Demand Zones (Driver)"])

# ---------------- TAB 1: CUSTOMER ----------------
with tab1:
    st.header(" Will This Ride Trigger Surge Pricing?")

    col1, col2 = st.columns(2)
    with col1:
        distance = st.slider("Ride Distance (km)", 1, 30, 5)
        base_fare = st.slider("Base Fare (₹)", 50, 300, 120)
        demand_index = st.slider("Demand Index", 1, 10, 5)
    with col2:
        hour = st.slider("Hour of Day", 0, 23, 18)
        day = st.selectbox("Day of Week", ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
        weather = st.selectbox("Weather Condition", df['weather_condition'].unique())
        traffic = st.selectbox("Traffic Level", df['traffic_level'].unique())

    # Encode inputs
    day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    weather_code = df[df['weather_condition'] == weather]['weather_encoded'].iloc[0]
    traffic_code = df[df['traffic_level'] == traffic]['traffic_encoded'].iloc[0]

    X_new = pd.DataFrame([{
        'ride_distance_km': distance,
        'base_fare': base_fare,
        'demand_index': demand_index,
        'hour': hour,
        'dayofweek': day_map[day],
        'weather_encoded': weather_code,
        'traffic_encoded': traffic_code
    }])

    # Prediction
    prediction = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f" Surge Expected! (Probability: {proba:.2f})")
    else:
        st.info(f" No Surge Likely (Probability: {proba:.2f})")

# ---------------- TAB 2: DRIVER ----------------
with tab2:
    st.header(" Driver Demand Analysis")

    # Top pickup zones
    top_locations = df['pickup_location'].value_counts().head(10)

    st.subheader(" Top 10 High-Demand Pickup Zones")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_locations.index, y=top_locations.values, ax=ax1)
    ax1.set_ylabel("Number of Rides")
    ax1.set_title("Top Pickup Locations")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    st.subheader(" Hourly Demand Heatmap for Top Zones")
    # FIXED: Count rides per zone per hour
    pivot = df.pivot_table(index='pickup_location', columns='hour', aggfunc='size')
    top10_pivot = pivot.loc[top_locations.index]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(top10_pivot.fillna(0), cmap="YlGnBu", annot=True, fmt=".0f", ax=ax2)
    ax2.set_title("Hourly Ride Demand per Pickup Zone")
    st.pyplot(fig2)