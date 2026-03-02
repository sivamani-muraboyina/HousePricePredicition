import streamlit as st
import joblib
import numpy as np

# 1. Load the model
model = joblib.load('models/house_price_model.pkl')

st.title("🏠 California House Price Predictor")
st.write("Adjust the sliders to see how they affect the house price.")

# 2. Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    med_inc = st.slider("Median Income (in $10k)", 0.5, 15.0, 3.5)
    house_age = st.slider("House Age", 1, 52, 28)
    ave_rooms = st.slider("Average Rooms", 1, 10, 5)

with col2:
    population = st.number_input("Neighborhood Population", value=1400)
    lat = st.number_input("Latitude", value=34.0)
    lon = st.number_input("Longitude", value=-118.0)

# 3. Prediction Logic
if st.button("Calculate Estimated Value"):
    # Arrange inputs in the exact order the model expects
    # (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Lon)
    features = np.array([[med_inc, house_age, ave_rooms, 1.0, population, 3.0, lat, lon]])
    prediction = model.predict(features)
    
    final_price = prediction[0] * 100000
    st.metric(label="Estimated Price", value=f"${final_price:,.2f}")