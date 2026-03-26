import streamlit as st
import joblib
import numpy as np

# Set page title
st.set_page_config(page_title="BPC Load Estimator", page_icon="⚡")

st.title("⚡ BPC Residential Load Estimator")
st.write("Enter the property size to predict expected monthly electricity consumption.")

# 1. Load the model (Make sure this file is uploaded to the same GitHub repo!)
try:
    model = joblib.load('electricity_model.pkl')
    
    # 2. User Input
    house_size = st.number_input("Property Size (Square Footage)", min_value=400, max_value=10000, value=1500)

    # 3. Prediction Button
    if st.button("Predict Consumption"):
        prediction = model.predict(np.array([[house_size]]))
        st.success(f"Estimated Monthly Consumption: {prediction[0]:.2f} kWh")
        st.info("This estimate helps BPC planners allocate transformer capacity efficiently.")

except FileNotFoundError:
    st.error("Error: 'electricity_model.pkl' not found in the repository. Please upload it!")

st.markdown("---")
st.caption("BPC Project 1 - Data Science Technician Portfolio")
