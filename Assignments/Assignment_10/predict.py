import os
import sys
import joblib
import pandas as pd

print("==================================================")
print("Automotive MLOps Model Inference Service Running")
print("==================================================")

model_path = 'rf_model.joblib'
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(__file__), 'rf_model.joblib')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model artifact '{model_path}' not found. Train and export model first.")

model = joblib.load(model_path)
print(f"Successfully loaded trained model artifact from '{model_path}'.")

sample_input = pd.DataFrame([{
    'age_years': 3.0,
    'mileage_km': 35000.0,
    'engine_size_L': 2.0,
    'horsepower': 180.0
}])

print(f"\nSample Input Features:\n{sample_input.to_string(index=False)}")

predicted_price = model.predict(sample_input)[0]
print(f"\nActual Model Predicted Resale Price: ${predicted_price:.2f}")
print("==================================================")
print("Inference service execution completed successfully.")
