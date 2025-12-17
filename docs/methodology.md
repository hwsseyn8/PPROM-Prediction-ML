# Methodology

## Study Design

This study implements a machine learning pipeline for predicting Preterm Premature Rupture of Membranes (PPROM) using various classification algorithms.

## Data Preprocessing

### 1. Data Cleaning
- Removal of rows with >50% missing values
- Handling of categorical variables with safe encoding
- KNN imputation for missing values

### 2. Feature Scaling
- StandardScaler: Standardization (mean=0, std=1)
- MaxAbsScaler: Scales by maximum absolute value
- Option to skip scaling for tree-based models

### 3. Train-Test Split
- Stratified split to maintain class distribution
- 80-20 split by default (configurable)

## Feature Engineering

### 1. Feature Selection
- **Boruta**: All-relevant feature selection
- **Lasso**: Sparse feature selection with regularization
- **RFE**: Recursive feature elimination
- **None**: Use all features

### 2. Dimensionality Reduction
- **PCA**: Principal Component Analysis
- Explained variance threshold: 95% by default

## Class Imbalance Handling

### 1. Sampling Techniques
- **SMOTE**: Synthetic Minority Over-sampling
- **ADASYN**: Adaptive Synthetic Sampling
- **Random Oversampling**: Duplicate minority samples
- **Random Undersampling**: Remove majority samples
- **Combined Methods**: SMOTEENN, SMOTETomek

### 2. Algorithm-Level Methods
- Class weighting in models
- Cost-sensitive learning

## Model Training

### 1. Algorithms Implemented
- Logistic Regression (with regularization)
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines
- Neural Networks (MLP)
- Ensemble methods (Voting, Stacking)

### 2. Hyperparameter Tuning
- **Random Search**: Efficient exploration of parameter space
- **Bayesian Optimization**: Optuna-based intelligent search
- Cross-validation for performance estimation

### 3. Model Calibration
- **Isotonic Regression**: Non-parametric calibration
- **Platt Scaling**: Logistic regression calibration
- **Sigmoid**: Parametric calibration

## Evaluation Metrics

### 1. Primary Metrics
- Accuracy, Precision, Recall, Specificity
- F1-Score (harmonic mean of precision and recall)
- ROC AUC (Area Under ROC Curve)
- Average Precision (Area Under PR Curve)

### 2. Advanced Metrics
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy
- Negative Predictive Value (NPV)
- Brier Score (calibration)
- Expected Calibration Error (ECE)

### 3. Statistical Analysis
- Confidence intervals via bootstrapping
- Friedman test for multiple comparisons
- Wilcoxon signed-rank test for pairwise comparisons

## Model Interpretation

### 1. Feature Importance
- Tree-based importance scores
- SHAP (SHapley Additive exPlanations) values
- Partial dependence plots

### 2. Clinical Interpretability
- Risk stratification thresholds
- Decision curve analysis
- Nomogram development

## Validation Strategy

### 1. Internal Validation
- 5-fold cross-validation
- Stratified sampling
- Nested cross-validation for hyperparameter tuning

### 2. External Validation
- Hold-out test set (20%)
- Temporal validation (if data available)
- Geographic validation (if multi-center data)

## Software Implementation

### 1. Programming Language
- Python 3.10+
- Scikit-learn ecosystem
- Specialized libraries (XGBoost, LightGBM, SHAP)

### 2. Code Organization
- Modular design for reproducibility
- Configuration management
- Comprehensive documentation
- Unit testing

## Ethical Considerations

### 1. Data Privacy
- De-identified data only
- Compliance with HIPAA/GDPR
- Institutional review board approval

### 2. Clinical Applicability
- Risk-benefit analysis
- Clinical utility assessment
- Implementation considerations

## Limitations

### 1. Methodological Limitations
- Single-center data (if applicable)
- Retrospective design
- Potential unmeasured confounding

### 2. Technical Limitations
- Computational requirements
- Model interpretability trade-offs
- Generalizability concerns

## Future Directions

### 1. Methodological Improvements
- Deep learning approaches
- Transfer learning
- Multi-task learning

### 2. Clinical Integration
- Real-time prediction
- Mobile application
- Electronic health record integration
