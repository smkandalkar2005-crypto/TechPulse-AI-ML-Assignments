# Assignment 9 — Feature Importance Visualization & Interpretation

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Quantify, visualize, and interpret the feature importances of an automotive resale pricing dataset using **Random Forest Mean Decrease in Impurity (MDI)** feature importances and test set **Permutation Importance**.

## 2. Dataset Description
- **Records:** Exactly 600 synthetic vehicle records.
- **Features:**
  - `age_years`: Vehicle age in years (1.1 to 15.0 years).
  - `mileage_km`: Odometer reading in kilometers (12,007 to 250,000 km).
  - `engine_size_L`: Engine displacement in liters (1.01 to 4.49 L).
  - `horsepower`: Engine power in HP (49.0 to 395.0 HP).
  - `fuel_efficiency_kmpl`: Fuel efficiency in km/L (8.6 to 29.1 km/L).
  - `vehicle_weight_kg`: Curb weight in kg (1,220 to 2,461 kg).
  - `brand`: One-Hot Encoded vehicle brand (`Toyota`, `Honda`, `Ford`, `BMW`, `Hyundai`).
- **Target:** `price` (Vehicle resale price in USD).

## 3. Model & Evaluation Performance
- **Model:** `RandomForestRegressor(n_estimators=300, random_state=42)`
- **Train/Test Split:** 80/20 train-test split (`random_state=42`).
- **Regression Evaluation Metrics:**
  - **Mean Absolute Error (MAE):** **\$2051.48**
  - **Root Mean Squared Error (RMSE):** **\$2690.46**
  - **Coefficient of Determination ($R^2$):** **0.9662**

## 4. Actual Executed Feature Importances
All values below come directly from the executed notebook (`TTL_Assignment_09.ipynb`):

### Random Forest Mean Decrease in Impurity (MDI)
1. `mileage_km`: **0.4064** (40.64%)
2. `age_years`: **0.3876** (38.76%)
3. `horsepower`: **0.1066** (10.66%)
4. `brand_BMW`: **0.0379** (3.79%)
5. `engine_size_L`: **0.0234** (2.34%)
6. `vehicle_weight_kg`: **0.0220** (2.20%)
7. `fuel_efficiency_kmpl`: **0.0122** (1.22%)

### Out-of-Sample Permutation Importance (Test Set)
1. `age_years`: **0.4517** (45.17% $R^2$ decrease when shuffled)
2. `mileage_km`: **0.3854** (38.54% $R^2$ decrease)
3. `horsepower`: **0.1157** (11.57% $R^2$ decrease)
4. `brand_BMW`: **0.0886** (8.86% $R^2$ decrease)

## 5. Key Visualizations Included
1. **Correlation Matrix Heatmap:** Linear relationships between numeric features and resale price.
2. **Side-by-Side Horizontal Bar Chart:** Random Forest Mean Decrease in Impurity (MDI) vs. Test Set Permutation Importance.

## 6. Interpretation & Causation Disclaimer
- `age_years` and `mileage_km` contribute approximately **79% of total MDI feature importance** (40.64% + 38.76%).
- `horsepower` and luxury brand tier (`brand_BMW`) serve as key secondary price drivers.
- **Causation Disclaimer:** Feature importance scores reflect the model's learned dependence on features within this specific dataset and do **not** constitute mathematical proof of direct physical or market causation.

## 7. How to Run
1. Navigate to `Assignments/Assignment_09/`:
   ```bash
   cd Assignments/Assignment_09
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_09.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).
