import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('best_gdm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'best_gdm_model.pkl' is in the same directory.")
        return None

model = load_model()

# Define the path for storing patient data
PATIENT_DATA_FILE = 'patient_data.csv'

# Load or initialize patient data
@st.cache_data
def load_patient_data():
    if os.path.exists(PATIENT_DATA_FILE):
        return pd.read_csv(PATIENT_DATA_FILE)
    else:
        return pd.DataFrame(columns=['Age', 'Fasting Glucose', 'Fasting Insulin ', 'Fasting C-Peptide',
                                     'Fasting Leptin', 'Fasting TG', 'Fasting Cho', 'HDL -C', 'LDL-C',
                                     'VLDL', 'GDM Prediction'])

patient_df = load_patient_data()

# Function to save patient data
def save_patient_data(df_to_save):
    df_to_save.to_csv(PATIENT_DATA_FILE, index=False)

# Web Application Interface
st.title("Gestational Diabetes Mellitus (GDM) Prediction")

st.write("Enter the patient's medical details to get a GDM prediction.")

# Get the list of input features (excluding the target and dropped features)
input_features = ['Age', 'Fasting Glucose', 'Fasting Insulin ', 'Fasting C-Peptide',
                  'Fasting Leptin', 'Fasting TG', 'Fasting Cho', 'HDL -C', 'LDL-C', 'VLDL']

# Create input fields dynamically
input_data = {}
st.sidebar.header("Patient Input")
for feature in input_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)

# Display entered values
st.subheader("Entered Values:")
entered_values_df = pd.DataFrame(input_data, index=["Value"])
st.write(entered_values_df)

if st.sidebar.button("Predict GDM"):
    if model is not None:
        global patient_df # Moved global declaration to the top of the block

        # Prepare data for prediction
        input_df = pd.DataFrame([input_data])

        # Ensure columns are in the same order as training data and handle potential missing columns
        # Note: In a real application, you would need to handle missing values in the input data as well
        # For this example, we assume all input features are provided.
        # If not, you would need to apply the same imputation/scaling as during training.
        # For simplicity here, we just ensure column order.
        input_df = input_df[input_features]


        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"Based on the inputs, the model predicts the patient **has GDM**.")
            st.write(f"Confidence (Probability of GDM): {prediction_proba[0]:.2f}")
            prediction_result = "GDM"
        else:
            st.success(f"Based on the inputs, the model predicts the patient **does not have GDM**.")
            st.write(f"Confidence (Probability of Not GDM): {1 - prediction_proba[0]:.2f}")
            prediction_result = "No GDM"

        # Store the result
        new_patient_data = input_data.copy()
        new_patient_data['GDM Prediction'] = prediction_result
        patient_df = pd.concat([patient_df, pd.DataFrame([new_patient_data])], ignore_index=True)
        save_patient_data(patient_df)
        st.success("Patient data and prediction saved.")
    else:
        st.warning("Model not loaded. Cannot make prediction.")


st.subheader("Previous Patient Results:")
if not patient_df.empty:
    st.dataframe(patient_df)

    # Download option
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(patient_df)
    st.download_button(
        label="Download Previous Results as CSV",
        data=csv,
        file_name='gdm_patient_results.csv',
        mime='text/csv',
    )
else:
    st.write("No previous patient results yet.")
