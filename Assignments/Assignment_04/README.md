# Assignment 4 — Vehicle Price Prediction

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Estimate vehicle resale price from structured vehicle data containing numerical and categorical features using regression models (**Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**).

## 2. Dataset Description
- **Records:** Exactly 600 synthetic vehicle records.
- **Categorical Features:** `brand` (`Toyota`, `Honda`, `Ford`, `BMW`, `Hyundai`), `fuel_type` (`Petrol`, `Diesel`, `Electric`).
- **Numerical Features:** `age_years`, `mileage_km`, `engine_size_L`.
- **Target Variable:** `price` (Vehicle resale price in USD).

## 3. Preprocessing & Pipeline Approach
- **`ColumnTransformer`:**
  - `StandardScaler()` applied to numerical features (`age_years`, `mileage_km`, `engine_size_L`).
  - `OneHotEncoder(handle_unknown='ignore')` applied to categorical features (`brand`, `fuel_type`).
- **`Pipeline` Encapsulation:** Encapsulated preprocessor and regressor within scikit-learn pipelines to prevent data leakage from test split.

## 4. Model Performance Comparison
- **Linear Regression:** MAE = $1847.62 | RMSE = $2250.87 | $R^2$ = 0.9505
- **Gradient Boosting Regressor:** MAE = $1947.18 | RMSE = $2446.63 | $R^2$ = 0.9415
- **Random Forest Regressor:** MAE = $2264.45 | RMSE = $2746.79 | $R^2$ = 0.9263

**Best Model:** **Linear Regression** (Lowest RMSE = $2250.87).

## 5. How to Run
1. Navigate to `Assignments/Assignment_04/`:
   ```bash
   cd Assignments/Assignment_04
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_04.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).

## 6. Conclusion
Linear Regression provided the highest predictive precision on this structured vehicle pricing dataset, demonstrating that linear relationships between age, mileage, brand tier, and price are effectively captured via standardized pipelines.
