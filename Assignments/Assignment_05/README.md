# Assignment 5 — Predictive Maintenance from Sensor Logs

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Classify whether a vehicle or machine component is at risk of failure using simulated sensor log data and a **Random Forest Classifier**.

## 2. Dataset Description
- **Records:** Exactly 800 synthetic sensor log records.
- **Features:**
  - `temperature_C`: Operating temperature in °C.
  - `vibration_mm_s`: Vibration velocity in mm/s.
  - `pressure_psi`: Fluid/gas pressure in PSI.
  - `rpm`: Engine/motor rotation speed in RPM.
  - `run_hours`: Total operational run hours.
- **Target:** `failure` (`0 = Healthy`, `1 = Failure`). Imbalanced class distribution (~25.6% failure rate).

## 3. Preprocessing & Model Configuration
- **Train/Test Split:** Stratified 75/25 split (`stratify=y`, `random_state=42`) to preserve class imbalance ratio.
- **Standardization:** `StandardScaler` fit strictly on `X_train` to prevent data leakage.
- **Model:** `RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')`.

## 4. Performance Summary
- **Healthy Class (0):** Precision = 0.82 | Recall = 0.95 | F1-Score = 0.88
- **Failure Class (1):** Precision = 0.73 | Recall = 0.37 | F1-Score = 0.49
- **Confusion Matrix:** TN = 142 | FP = 7 | FN = 32 | TP = 19

## 5. How to Run
1. Navigate to `Assignments/Assignment_05/`:
   ```bash
   cd Assignments/Assignment_05
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_05.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).

## 6. Conclusion
The Random Forest Classifier successfully balances decision thresholds using `class_weight='balanced'`, providing robust healthy component recognition while detecting component failure risks from operational sensor signatures.
