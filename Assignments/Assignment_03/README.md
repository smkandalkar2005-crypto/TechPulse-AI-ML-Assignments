# Assignment 3 — Data Cleaning & Preprocessing Techniques

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Demonstrate essential data cleaning and preprocessing techniques—including missing value analysis and median/mode imputation, IQR outlier detection and capping, nominal one-hot encoding, and feature standardization—using Pandas and `scikit-learn`.

*Note: This assignment focuses strictly on Data Preprocessing. Machine learning model training is intentionally omitted.*

## 2. Dataset Description
- **Records:** 120 synthetic automotive data records.
- **Features:**
  - `mileage_km`: Vehicle mileage (numeric, contains NaNs & outliers).
  - `age_years`: Vehicle age in years (numeric, contains NaNs).
  - `engine_size_L`: Engine displacement in liters (numeric).
  - `price`: Vehicle price in USD (numeric, contains NaNs & extreme price outliers).
  - `fuel_type`: Categorical fuel category (`Petrol`, `Diesel`, `Electric`, contains NaNs).

## 3. Preprocessing Pipeline Steps
1. **Missing Value Imputation:**
   - Median imputation (`SimpleImputer(strategy='median')`) for numeric columns.
   - Most frequent imputation (`SimpleImputer(strategy='most_frequent')`) for `fuel_type`.
2. **IQR Outlier Capping:**
   - Upper/Lower bounds set to $Q_1 - 1.5 	imes IQR$ and $Q_3 + 1.5 	imes IQR$.
   - Winsorization (`np.clip`) applied to cap outliers without discarding rows.
3. **One-Hot Encoding:**
   - `OneHotEncoder(drop='first', sparse_output=False)` applied to `fuel_type`.
4. **Standardization:**
   - `StandardScaler()` applied to scale numeric attributes to $\mu=0, \sigma=1$.
5. **Final Model-Ready DataFrame (`processed_df`):**
   - Concatenated scaled numeric features and encoded categorical columns into a clean 120 × 6 numeric matrix.

## 4. How to Run
1. Open terminal and navigate to `Assignments/Assignment_03/`:
   ```bash
   cd Assignments/Assignment_03
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_03.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).

## 5. Expected Output
- Initial raw data missingness and statistics report.
- Zero missing values post-imputation verification.
- IQR bounds computation and outlier count summary.
- Before vs. After outlier treatment box plot comparisons.
- Final model-ready `processed_df` verification (0 NaNs, 100% numeric features).

## 6. Conclusion
The dataset was transformed from an incomplete, outlier-contaminated state into a clean, normalized, model-ready format ready for machine learning algorithms.
