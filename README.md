# PPROM Prediction using Machine Learning Models

This repository contains a comprehensive machine learning pipeline for predicting Preterm Premature Rupture of Membranes (PPROM) using various classification models.

## Overview

This study implements a complete machine learning workflow for PPROM prediction, including:
- Data preprocessing and feature engineering
- Multiple class imbalance handling techniques
- Extensive model training with hyperparameter tuning
- Model calibration and ensemble methods
- Comprehensive evaluation and visualization

## Project Structure
```bash
pprom-prediction-study/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/
├── notebooks/
├── scripts/
├── data/
├── models/
├── results/
├── tests/
└── docs/
```
## Quick Start

First Install dependencies:
pip install -r requirements.txt
Prepare your data in data/raw/data23.csv

Then
  Run the pipeline:
  python scripts/run_pipeline.py

Or 
  Run individual components:
  python scripts/train_model.py
  python scripts/evaluate_model.py

## Key Features

- Multiple Models: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, etc.
- Class Imbalance Handling: SMOTE, ADASYN, Random Oversampling/Undersampling
- Feature Selection: Boruta, Lasso, RFE, PCA
- Model Calibration: Isotonic, Sigmoid, Platt scaling
- Comprehensive Evaluation: ROC curves, precision-recall curves, calibration plots
- SHAP Interpretation: Model interpretability using SHAP values

## Results

Results are saved in the results/ directory including:
- Model performance metrics
- Feature importance plots
- Calibration curves
- Confusion matrices

## Citation

