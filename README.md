# Age Prediction Based on Gut Microbiome Profiles

## Objective
To develop machine learning models to predict an individual's age using gut microbiome profiles. This project identifies microbial community composition patterns linked to age and optimizes predictive performance.

## Tools and Libraries
- **Data Manipulation**: `pandas`, `dplyr`
- **Data Visualization**: `matplotlib`, `seaborn`, `ggplot2`
- **Data Preprocessing**: `nearZeroVar` (`caret`), `StandardScaler`
- **Feature Selection**: `Recursive Feature Elimination (RFE)`
- **Model Development**:
  - **Regression Models**: `RandomForestRegressor`, `XGBoost`, `Support Vector Regression (SVR)`
- **Model Evaluation**: `RMSE`, `MAE`, `R-squared`
- **Cross-Validation**: `trainControl` (`caret`)
- **Utility**: `gridExtra`, `skimr`

## Dataset
- **Taxonomy Data**: Composition of microbial communities for each sample.
- **Age Data**: Target variable, `Age`, associated with sample IDs.
- **Details**: Combined dataset includes 60 samples and 5656 features, reduced to 373 features following preprocessing.

## Methodology
1. **Data Preprocessing**:
   - Removed near-zero variance predictors.
   - Eliminated highly correlated features.
   - Scaled features to standardize ranges.
2. **Feature Selection**:
   - Applied Recursive Feature Elimination (RFE) with Random Forest to identify the top 10 most relevant features for predicting age.
3. **Model Development**:
   - Trained models using `RandomForestRegressor`, `XGBoost`, and `SVR`.
   - Conducted hyperparameter tuning for each model to optimize performance.

## Models
| Model             | RMSE   | MAE   | R-squared |
|--------------------|--------|-------|-----------|
| Random Forest      | 10.79  | 8.85  | 0.17      |
| XGBoost            | 10.32  | 8.57  | 0.24 (Best Performance) |
| Support Vector Regression (SVR) | 11.27  | 9.29  | 0.09      |

## Conclusion
The XGBoost model showed the highest predictive capability, with:
- **RMSE**: 10.32
- **MAE**: 8.57
- **R-squared**: 0.24

This study highlights the potential of gut microbiome data for age-related health assessments. Further refinement is recommended to improve accuracy, particularly for extreme age values.
