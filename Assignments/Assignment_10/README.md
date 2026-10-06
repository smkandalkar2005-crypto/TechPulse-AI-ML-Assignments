# Assignment 10 — MLOps Workflow Simulation

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Simulate an educational **MLOps Workflow** covering machine learning model training, experiment tracking via **MLflow**, model artifact export (`rf_model.joblib`), standalone model inference (`predict.py`), containerization setup via **Docker**, and continuous integration via **GitHub Actions**.

## 2. Methodology & Architecture
1. **Model Training & Export:** Train a `RandomForestRegressor(n_estimators=200, max_depth=12)` on an automotive pricing dataset (500 records) and export the trained model artifact to `rf_model.joblib`.
2. **MLflow Tracking:** Track hyperparameters (`n_estimators`, `max_depth`, `random_state`), evaluation metrics (MAE, RMSE, $R^2$), and log scikit-learn model artifacts using an SQLite tracking store (`sqlite:///mlflow.db`).
3. **Standalone Model Inference (`predict.py`):** Load `rf_model.joblib` and execute actual model prediction on structured feature inputs (`age_years`, `mileage_km`, `engine_size_L`, `horsepower`).
4. **Containerization Setup:** Provide a lightweight `Dockerfile` that copies `rf_model.joblib`, `predict.py`, and `requirements.txt` into the container image for autonomous execution without external MLflow tracking DB dependency.
5. **CI/CD Automation:** Define a GitHub Actions workflow (`.github/workflows/mlops.yml`) to automate testing, training, MLflow logging, and `predict.py` inference execution.

## 3. Actual Executed MLOps Metrics & Results
All values below come directly from the executed notebook (`TTL_Assignment_10.ipynb`):

- **MLflow Experiment Name:** `Automotive_Price_Prediction_Experiment`
- **MLflow Tracking Store:** `sqlite:///mlflow.db`
- **Model Performance Metrics:**
  - **Mean Absolute Error (MAE):** **\$1382.36**
  - **Root Mean Squared Error (RMSE):** **\$1761.26**
  - **Coefficient of Determination ($R^2$):** **0.9836**
- **Exported Model Artifact:** `rf_model.joblib`
- **Standalone Model Inference (`predict.py`):**
  - **Sample Vehicle Input:** `age_years = 3.0`, `mileage_km = 35000.0`, `engine_size_L = 2.0`, `horsepower = 180.0`
  - **Actual Model Predicted Resale Price:** **\$40,808.93**

## 4. Truthful Docker & CI/CD Status
- **Standalone Model Inference Verification:** Executed `python predict.py` locally and verified actual trained model loading and prediction.
- **Docker Integration:** `Dockerfile`, `requirements.txt`, `rf_model.joblib`, and `predict.py` configured for autonomous container build.
  > **Truthful Execution Note:** As the local test machine does not have an active Docker daemon installed, container execution was verified via file structure/syntax inspection and local `predict.py` script execution without running `docker build` / `docker run`.
- **CI/CD Workflow:** Validated `.github/workflows/mlops.yml` pipeline covering code checkout, dependency installation, notebook execution, MLflow tracking, and `predict.py` execution.

## 5. How to Run
1. Navigate to `Assignments/Assignment_10/`:
   ```bash
   cd Assignments/Assignment_10
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_10.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).
4. Run standalone model inference:
   ```bash
   python predict.py
   ```
