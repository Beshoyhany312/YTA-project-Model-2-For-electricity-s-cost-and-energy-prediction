import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import joblib
import os

# --- PATH FIX ---
# This ensures the app finds files relative to THIS script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "multi_output_lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.joblib")

def load_model_and_scaler(model_path, scaler_path):
    """Loads the Keras model and the StandardScaler with error handling."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    
    # Load model (compile=False is good for inference)
    model = keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_input(new_data_df, scaler):
    processed_df = new_data_df.copy()

    # One-hot encoding logic
    if 'season' in processed_df.columns:
        processed_df['season_winter'] = (processed_df['season'].str.lower() == 'winter').astype(int)
        processed_df = processed_df.drop(columns=['season'])
    else:
        processed_df['season_winter'] = 0

    if 'insulation_quality' in processed_df.columns:
        processed_df['insulation_quality_low'] = (processed_df['insulation_quality'].str.lower() == 'low').astype(int)
        processed_df['insulation_quality_medium'] = (processed_df['insulation_quality'].str.lower() == 'medium').astype(int)
        processed_df = processed_df.drop(columns=['insulation_quality'])
    else:
        processed_df['insulation_quality_low'] = 0
        processed_df['insulation_quality_medium'] = 0

    expected_feature_columns = [
        'number_of_air_conditioners', 'ac_power_hp', 'number_of_refrigerators',
        'number_of_televisions', 'number_of_fans', 'number_of_computers',
        'average_daily_usage_hours', 'house_size_m2', 'has_water_heater',
        'washing_machine_usage_per_week', 'season_winter', 
        'insulation_quality_low', 'insulation_quality_medium'
    ]

    for col in expected_feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    processed_df = processed_df[expected_feature_columns]
    
    # Scaling
    scaled_data = scaler.transform(processed_df)
    
    # Reshape for LSTM: (samples, 1, features)
    return scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])

def predict_electricity(new_data_df, model, scaler):
    processed_data = preprocess_input(new_data_df, scaler)
    predictions = model.predict(processed_data)
    
    # Multi-output models return a list of arrays [output1, output2]
    kwh_pred = predictions[0]
    bill_pred = predictions[1]
    
    return kwh_pred.flatten(), bill_pred.flatten()

# --- MAIN BLOCK FOR STREAMLIT INTEGRATION ---
if __name__ == '__main__':
    try:
        multi_output_lstm_model, feature_scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
        print("Model and scaler loaded successfully.")
        
        # ... (Rest of your example data and prediction code)
    except Exception as e:
        print(f"Deployment Error: {e}")
