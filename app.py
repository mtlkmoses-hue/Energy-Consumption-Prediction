import streamlit as st
import joblib
import numpy as np

# 1. Page Title & Branding
st.set_page_config(page_title="BPC Energy Predictor", page_icon="⚡")
st.title("⚡ BPC Energy Consumption Predictor")
st.write("Predicting household electricity usage for Gaborone properties.")

# 2. Load the Model correctly using joblib
try:
    # Notice we don't use 'with open' for joblib, it's simpler!
    model = joblib.load('electricity_model.pkl')
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# 3. User Input (Square Footage)
st.subheader("Property Details")
sq_ft = st.number_input("Enter Square Footage of the House:", min_value=100, max_value=10000, value=1000)

# 4. Predict Button
if st.button("Calculate Estimated Usage"):
    # Reshape input to match model requirements
    input_data = np.array([[sq_ft]])
    prediction = model.predict(input_data)
    
    # Check if this is our Logistic (Classification) or Linear (Continuous) model
    # If the output is an integer 0 or 1, it's the Logistic model
    if hasattr(model, "predict_proba"): 
        category = "High Consumer" if prediction[0] == 1 else "Efficient Consumer"
        st.metric("Energy Category", category)
    else:
        st.metric("Estimated Monthly Usage", f"{prediction[0]:.2f} kWh")

st.markdown("---")
st.caption("BPC Project 1 - Data Science Technician Portfolio")
