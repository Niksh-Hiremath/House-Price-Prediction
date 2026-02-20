# House Price Prediction using Ensemble Regression

## Overview

This project implements a complete machine learning pipeline to predict residential property prices using structured housing data. The workflow includes exploratory data analysis (EDA), feature engineering, outlier handling, model benchmarking, hyperparameter tuning, and ensemble learning using Voting and Stacking regressors.

The final selected model is a Voting Regressor combining KNeighborsRegressor, RandomForestRegressor, and MLPRegressor.

---

## Problem Statement

Real estate price prediction is a regression problem influenced by multiple structural and geographical factors. The objective of this project is to build a robust, generalizable regression pipeline that:

- Cleans and preprocesses raw housing data
- Engineers meaningful features
- Benchmarks multiple regression models
- Selects top-performing models
- Builds ensemble models for improved performance

---

## Dataset

The dataset contains structured housing records with features such as:

- Location
- Total square footage
- Number of bedrooms (BHK)
- Number of bathrooms
- Availability status
- Price (target variable)

Preprocessing was applied to handle inconsistencies, missing values, and noise.

---

## Data Preprocessing

### Data Cleaning
- Extracted numeric BHK values from text fields
- Converted square footage to numeric format
- Handled missing values using median (numeric) and mode (categorical)
- Encoded availability as a binary feature

### Location Engineering
- Grouped rare locations into an "Other" category to reduce high cardinality
- Applied OneHotEncoder with `handle_unknown="ignore"`

### Outlier Removal
- Removed unrealistic entries using domain constraints (e.g., bathrooms > BHK + 2)
- Trimmed extreme values using percentile filtering (2nd–98th percentile)
- Removed properties with implausible price per square foot

### Feature Transformation
- Applied log transformation to skewed price distribution
- Standardized numerical features using StandardScaler
- Encoded categorical features using OneHotEncoder

All preprocessing steps were integrated into a Scikit-learn Pipeline using ColumnTransformer.

---

## Exploratory Data Analysis

EDA included:

- Distribution analysis of price and square footage
- Log-transformed target visualization
- Correlation heatmap
- Price vs. location analysis
- Size vs. price relationship
- Skewness evaluation

---

## Model Benchmarking

A total of 11 regression models were trained and evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- KNeighborsRegressor
- Support Vector Regressor
- MLPRegressor
- Bagging Regressor

Primary evaluation metric:
- R² Score

Models were compared using a train-validation split.

---

## Hyperparameter Tuning

Top-performing models were selected for optimization:

- KNeighborsRegressor (GridSearchCV)
- RandomForestRegressor (RandomizedSearchCV)
- MLPRegressor (RandomizedSearchCV)

Tuning configuration:
- 5-fold Cross Validation
- Parallelized where applicable

---

## Ensemble Learning

Two ensemble strategies were implemented:

### Voting Regressor
Combined:
- KNeighborsRegressor
- RandomForestRegressor
- MLPRegressor

### Stacking Regressor
Base models:
- KNN
- Random Forest
- MLP

Meta-learner:
- Linear Regression

The final selected model was the Voting Regressor based on validation performance.

---

## Final Model Performance

Final Model:
Voting Regressor (KNN + Random Forest + MLP)

Validation Performance:
$R^2 ≈ 0.77$

---

## Key Takeaways

- Structured preprocessing pipelines improve reproducibility
- Outlier removal significantly impacts regression stability
- Model benchmarking is essential before tuning
- Ensemble methods improve generalization
- Cross-validation provides robust performance estimation

---
