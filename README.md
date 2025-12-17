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

1. Install dependencies:
pip install -r requirements.txt
Prepare your data in data/raw/data23.csv

Run the pipeline:
python scripts/run_pipeline.py

Or run individual components:
python scripts/train_model.py
python scripts/evaluate_model.py
