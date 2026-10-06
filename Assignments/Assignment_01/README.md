# Assignment 1 — ML Model for Car Mileage Estimation

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Build and compare supervised machine learning regression models (**Linear Regression** and **Random Forest Regression**) to predict car mileage (MPG) from vehicle technical specifications using Python and `scikit-learn`.

## 2. Dataset
- **Type:** Synthetic automotive specification dataset (~500 records).
- **Location:** `data/car_mileage_dataset.csv`
- **Reproducibility:** Seeded with `np.random.seed(42)` to ensure exact reproducibility.
- **Physical Realism:** Grounded in vehicle power-to-weight and displacement dynamics where heavier/more powerful engines yield lower fuel efficiency while newer model years reflect enhanced fuel economy.

## 3. Features and Target
### Input Features ($X$):
- `cylinders`: Engine cylinders (4, 6, 8)
- `displacement`: Engine displacement in cubic inches (cu in)
- `horsepower`: Engine horsepower rating (HP)
- `weight`: Vehicle curb weight in pounds (lbs)
- `acceleration`: Acceleration time (0–60 mph in seconds)
- `model_year`: Model year (70 to 82, representing 1970–1982)

### Target Variable ($y$):
- `mpg`: Miles Per Gallon (continuous fuel efficiency measure)

## 4. Algorithms Used
1. **Linear Regression (with `StandardScaler`):**
   - Fits an Ordinary Least Squares (OLS) linear hyperplane on scaled feature inputs.
2. **Random Forest Regression:**
   - Ensemble of 100 decision trees (`RandomForestRegressor(n_estimators=100, random_state=42)`).

## 5. Evaluation Metrics
- **MAE (Mean Absolute Error):** Measures average magnitude of prediction errors in MPG.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction variances.
- **$R^2$ Score (Coefficient of Determination):** Quantifies the proportion of variance in MPG explained by the technical attributes.

## 6. Project Structure
```
Assignment_01/
├── README.md                   # Assignment documentation
├── TTL_Assignment_01.ipynb     # Main Jupyter notebook with code & outputs
├── requirements.txt            # Environment dependencies
├── data/
│   └── car_mileage_dataset.csv # 500-car synthetic dataset
└── outputs/
    ├── eda_mpg_distribution.png       # Target variable EDA distribution
    ├── eda_correlation_heatmap.png    # Feature correlation matrix plot
    ├── eda_features_vs_mpg.png        # Technical specs vs MPG subplots
    ├── model_comparison.csv           # Model performance comparison metrics
    ├── prediction_vs_actual.png       # Actual vs Predicted MPG scatter plots
    └── rf_feature_importance.png      # Random Forest feature importance bar chart
```

## 7. Requirements
- Python 3.9+
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## 8. How to Run
1. Open terminal and navigate to `Assignment_01/`:
   ```bash
   cd Assignment_01
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_01.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).
4. Alternatively, execute headless via `nbconvert`:
   ```bash
   python -m jupyter nbconvert --to notebook --execute TTL_Assignment_01.ipynb --output TTL_Assignment_01.ipynb
   ```

## 9. Expected Outputs
Running the notebook populates:
- **`data/car_mileage_dataset.csv`**: Synthetic dataset with 500 records.
- **`outputs/model_comparison.csv`**: Performance metrics comparison table.
- **`outputs/*.png`**: EDA distribution, heatmap, feature relationships, prediction fit, and feature importances.

## 10. Summary of Results
- **Linear Regression:** MAE = 1.8651 MPG, RMSE = 2.3413 MPG, $R^2$ = 0.9143
- **Random Forest Regression:** MAE = 1.9706 MPG, RMSE = 2.5538 MPG, $R^2$ = 0.8981
- **Conclusion:** On this dataset split, Linear Regression outperforms Random Forest Regression across all three metrics (lower MAE, lower RMSE, and higher $R^2$). Vehicle weight and displacement are the primary technical factors driving fuel consumption.
