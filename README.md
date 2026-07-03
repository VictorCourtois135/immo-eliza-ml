# 🏠 Immo Eliza - Machine Learning Price Prediction

## 📌 Description

This project is part of the Immo Eliza series. After scraping and 
cleaning a dataset of Belgian real estate listings, the goal here is 
to build a machine learning model capable of predicting property 
prices based on their characteristics (location, size, condition, 
energy performance, etc.).

Six regression models were trained and compared, from a simple linear 
baseline to more complex ensemble methods (Random Forest, Gradient 
Boosting, XGBoost), in order to identify the best-performing approach 
for this dataset.

## 🎯 Objectives

- Preprocess real-world, messy real estate data for machine learning
- Build a reusable preprocessing pipeline (fit on train, applied identically to test)
- Train and compare several regression models
- Evaluate model performance using appropriate metrics
- Detect and reduce overfitting through hyperparameter tuning

## 📂 Repository Structure

```
immo-eliza-ml/
├── training.py                  # Main training pipeline
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/                      # Model class definitions
│   ├── model_template.py        # ModelTemplate (shared metrics logic)
│   ├── linear_regression.py
│   ├── ridge.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── gradient_boosting.py
│   └── xgboost.py
│
├── models_trained/               # Saved trained models 
│   ├── LinReg.pkl
│   ├── Ridge.pkl
│   ├── DecisionTree.pkl
│   ├── RandForest.pkl
│   ├── GradBoost.pkl
│   └── XGBoost.pkl
│
├── dataset/
│   └── properties_final_irene.csv
│
└── utils/
    ├── cleaning.py               # Missing values handling, column dropping
    ├── ordinal_encoding.py       # EPC / building state ordinal encoding
    ├── feature_engineering.py    # Feature creation, outlier filtering
    ├── onehot_encoding.py        # One-hot encoding of categorical features
    ├── persistence.py            # Model saving
    └── metrics.py                # Model evaluation & results printing
```

## ⚙️ Installation

```bash
git clone https://github.com/VictorCourtois135/immo-eliza-ml.git
cd immo-eliza-ml

python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

## 🚀 Usage

Train all models and print the comparison report:

```bash
python training.py
```

This will:
1. Load and clean the raw dataset
2. Split it into train/test sets
3. Fit preprocessing statistics on the training set only
4. Apply preprocessing consistently to both sets
5. Train 6 models
6. Save trained models 
7. Print a full metrics comparison table

## 🧠 Methodology

### 1. Data Cleaning

- Dropped irrelevant, redundant, or leaky columns (e.g. `address`, 
  `price_per_m2`, columns with excessive missing values)
- Dropped rows with missing `latitude`/`longitude` 
- **The dataset was split into train/test BEFORE any statistic (median, 
  category list) was computed**, to strictly avoid data leakage

### 2. Handling Missing Values (Imputation)

- Numerical features (`living_area_m2`, `bedrooms`, `bathrooms`, 
  `facades`, `building_year`) were imputed using the **median grouped 
  by `property_subtype`**, computed on the training set only, with a 
  fallback to the global training median for unseen subtypes
- `garden_area_m2` was set to 0 when the property has no garden, 
  otherwise imputed using group medians
- Categorical features (`epc_score`, `state_of_the_building`) were 
  filled with an explicit `"Unknown"` category
- For every imputed column, a binary `<column>_was_missing` flag was 
  added, allowing the model to learn from missingness itself

### 3. Outlier Filtering

- Extreme price outliers were removed using quantile thresholds 
  (1st and 98th percentile), computed on the training set and applied 
  identically to the test set

### 4. Ordinal Encoding

- `epc_score` (G to A++) and `state_of_the_building` (To demolish to 
  New) were mapped to ordered numeric scales, since these categories 
  have a natural ranking that one-hot encoding would discard

### 5. Feature Engineering

Several engineered features were added to help models capture 
non-linear relationships (23 new features):
- Interaction terms (e.g. `area_x_epc`, `area_x_state`, `quality_x_area`)
- Ratios (e.g. `area_per_bedroom`, `bed_bath_ratio`, `garden_to_living_ratio`)
- Composite quality scores combining EPC and building condition
- Non-linear transforms (`log`, `sqrt`) of area and quality features, 
  particularly useful for linear models
- Building age categories (`is_new_building`, `is_old_building`)

### 6. One-Hot Encoding

- Categorical features (`property_type`, `property_subtype`, 
  `region`, `province`) were one-hot encoded
- Categories were extracted from the training set only; any category 
  unseen in the test set is safely encoded as all-zero (baseline)

### 7. Target Transformation

- The target variable (`price`) was log-transformed (`log1p`) before 
  training, to reduce the impact of price skewness and stabilize 
  variance. Predictions are converted back to real prices using 
  `expm1` before computing evaluation metrics

### 8. Feature Scaling

- `StandardScaler` was applied for models sensitive to feature scale 
  (Linear Regression, Ridge)
- Tree-based models (Decision Tree, Random Forest, Gradient Boosting, 
  XGBoost) do not require scaling and were trained on raw feature 
  values

### 9. Model Training & Hyperparameter Tuning

Six models were trained and compared:

| Model | Type |
|---|---|
| Linear Regression | Linear baseline |
| Ridge Regression | Regularized linear |
| Decision Tree | Non-linear |
| Random Forest | Ensemble (bagging) |
| Gradient Boosting | Ensemble (boosting) |
| XGBoost | Ensemble (boosting) |

Hyperparameters for each model were tuned using **GridSearchCV** with 
K-Fold cross-validation, optimizing for R².

### 10. Evaluation

Each model was evaluated on a held-out test set using R², adjusted R², 
cross-validation scores, an overfitting gap/ratio (train R² − test R²), 
and standard error metrics (MAE, Median AE, MAPE, MSE, RMSE, Max 
Error, Explained Variance).

## 📊 Results

| Metric | LinReg | Ridge | GradBoost | XGBoost | RandForest | DecisionTree |
|---|---|---|---|---|---|---|
| R² (test) | 0.7220 | 0.7208 | **0.8011** | 0.8009 | 0.7725 | 0.6734 |
| R² (train) | 0.7266 | 0.7251 | 0.8698 | 0.8676 | 0.9612 | 0.7890 |
| R² adjusted | 0.7161 | 0.7149 | 0.7969 | 0.7967 | 0.7676 | 0.6664 |
| OOB Score | N/A | N/A | N/A | N/A | 0.7676 | N/A |
| CV R² mean | 0.7034 | 0.7056 | 0.7931 | 0.7922 | 0.7670 | 0.6671 |
| CV R² std | 0.0510 | 0.0460 | 0.0205 | 0.0199 | 0.0192 | 0.0168 |
| Overfit gap | 0.0045 | 0.0043 | 0.0687 | 0.0667 | 0.1887 | 0.1156 |
| Overfit (%) | 0.62% | 0.60% | 7.90% | 7.69% | 19.63% | 14.65% |
| Explained Var. | 0.5865 | 0.5925 | 0.7460 | 0.7460 | 0.7107 | 0.6239 |
| MAE | €87,382 | €87,599 | **€72,477** | €72,398 | €76,771 | €92,794 |
| Median AE | €53,655 | €54,207 | €42,986 | **€42,887** | €43,626 | €57,416 |
| MAPE | 25.59% | 25.63% | 21.18% | **21.15%** | 22.28% | 27.98% |
| MSE | €22.31B | €22.02B | €13.68B | **€13.67B** | €15.81B | €20.23B |
| RMSE | €149,366 | €148,383 | €116,955 | **€116,939** | €125,746 | €142,238 |
| Max Error | €3,796,579 | €3,528,993 | €1,069,872 | **€1,060,896** | €1,024,613 | €1,094,594 |

*(Best score per metric in bold.)*

### Interpretation

**Gradient Boosting and XGBoost are the clear winners**, both reaching 
a test R² of ~0.80, a MAPE of ~21%, and an RMSE around €117k — 
substantially outperforming every other model on every error metric. 
The two are almost tied: XGBoost has a very slight edge on MAE, RMSE 
and Max Error, while Gradient Boosting has a marginally higher R². In 
practice, the two models are interchangeable in performance.

**Linear Regression and Ridge underperform clearly** (R² ≈ 0.72, MAPE 
≈ 25.6%), confirming that the relationship between property 
characteristics and price is not well captured by a linear model, 
despite the feature engineering (interaction terms, log/sqrt 
transforms) added specifically to help linear models. This is 
expected: real estate pricing involves complex, non-linear 
interactions (e.g. location, size, and condition don't combine 
additively) that tree-based models capture more naturally.

**Random Forest reaches a decent test R² (0.77) but overfits the most** 
of all six models (see Overfitting Analysis below), and its OOB score 
(0.7676) closely matches its CV/test R², confirming the overfitting 
diagnosis is consistent across independent validation methods.

**Decision Tree is the weakest non-linear model** (R² = 0.673, MAPE = 
28%), which is expected: a single tree, even regularized 
(`max_depth=9`, `min_samples_leaf=10`), has much less representational 
power than ensemble methods that combine hundreds of trees.

## 🔍 Overfitting Analysis

Overfitting was assessed by comparing R² on the training set vs. the 
test set (**overfit gap**), and by checking the consistency between 
cross-validation scores and test scores.

| Model | Overfit gap | Verdict |
|---|---|---|
| LinReg / Ridge | 0.60–0.62% | ✅ No overfitting — if anything, slightly underfitting (high bias, low variance) |
| XGBoost | 7.69% | ✅ Mild, well-controlled overfitting |
| GradBoost | 7.90% | ✅ Mild, well-controlled overfitting |
| Decision Tree | 14.65% | ⚠️ Moderate overfitting |
| Random Forest | 19.63% | ❌ Most overfit model |

- **Linear models (LinReg, Ridge) do not overfit** — their train and 
  test R² are nearly identical. However, their relatively low absolute 
  R² (~0.72) suggests they underfit the data: they are too simple to 
  capture the underlying patterns.
- **Random Forest shows the largest gap** (train R² = 0.96 vs. test R² 
  = 0.77), meaning it memorizes noise in the training data rather than 
  learning generalizable patterns. This is confirmed independently by 
  its OOB score (0.768), which is close to its test/CV R² — both 
  validation methods agree the model's true generalization capacity is 
  around 0.77, well below its inflated training score.
- **Gradient Boosting and XGBoost strike the best balance**: they 
  achieve the highest test R² while keeping the overfit gap moderate 
  (~7.7–7.9%), thanks to their built-in regularization (`subsample`, 
  `min_samples_leaf`/`min_child_weight`, shallow trees via 
  `max_depth=4`) tuned via GridSearchCV.
- **Cross-validation std** is also informative: LinReg/Ridge show the 
  highest CV std (~0.05), meaning their performance is more sensitive 
  to which data points end up in each fold — another sign of an 
  unstable, poorly-fitting model on this dataset.

**Conclusion:** Gradient Boosting and XGBoost are selected as the 
final models, offering the best trade-off between predictive accuracy 
and generalization.


## 👤 Author

Victor Courtois         
Github link: https://github.com/VictorCourtois135

## 📅 Timeline

This project was completed in 5 days as part of the BeCode AI/Data 
Science bootcamp.