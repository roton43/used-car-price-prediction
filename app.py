import streamlit as st
import pandas as pd
import joblib
import os

# Load the pipeline ONCE at startup (not inside a function)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
pipeline = joblib.load(MODEL_PATH)

# Page configuration
st.set_page_config(
    page_title='Used Car Price Predictor',
    page_icon='🚗',
    layout='centered'
)

# Header
st.title('🚗 Used Car Price Predictor')
st.markdown(
    'Estimate the **resale value** of a used car based on its specifications. '
    'Fill in the details below and click **Predict Selling Price**.'
)
st.divider()

# Input widgets — must exactly match training column names & categories

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input(
        'Present Price (in Lakhs)',
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1,
        help='Current ex-showroom price of the car model'
    )
    kms_driven = st.number_input(
        'Kilometres Driven',
        min_value=0,
        max_value=500000,
        value=30000,
        step=100,
        help='Total kilometres the car has been driven'
    )
    car_age = st.slider(
        'Car Age (years)',
        min_value=1,
        max_value=20,
        value=5,
        help='Number of years since the car was manufactured'
    )

with col2:
    fuel_type = st.selectbox(
        'Fuel Type',
        options=['Petrol', 'Diesel', 'CNG'],
        help='Type of fuel the car uses'
    )
    seller_type = st.selectbox(
        'Seller Type',
        options=['Dealer', 'Individual'],
        help='Whether the seller is a dealer or an individual owner'
    )
    transmission = st.selectbox(
        'Transmission',
        options=['Manual', 'Automatic'],
        help='Type of transmission'
    )
    owner = st.selectbox(
        'Number of Previous Owners',
        options=[0, 1, 2, 3],
        help='How many owners the car has had before the current one'
    )

st.divider()

# Prediction
if st.button('🔍 Predict Selling Price', type='primary', use_container_width=True):
    input_df = pd.DataFrame([{
        'Present_Price': present_price,
        'Kms_Driven'  : kms_driven,
        'Car_Age'     : car_age,
        'Fuel_Type'   : fuel_type,
        'Seller_Type' : seller_type,
        'Transmission': transmission,
        'Owner'       : owner,
    }])

    prediction = pipeline.predict(input_df)[0]
    prediction = max(prediction, 0.01)  

    st.success(f'### 💰 Estimated Selling Price: TK {prediction:.2f} Lakhs')

    # Show input summary
    with st.expander('📋 Input Summary'):
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

# Footer info
st.divider()
st.markdown(
    '<small>**Dataset:** CarDekho Used Cars (Kaggle) · '
    '**Model:** Linear Regression · '
    '**Test R²:** 0.9861 · '
    '**RMSE:** 0.49 Lakhs · '
    'Prepared by Md. Roton Ahmed</small>',
    unsafe_allow_html=True
)
