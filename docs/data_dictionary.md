# Data Dictionary

## Overview

This document describes the data structure and variables used in the PPROM prediction study.

## Data File

**File**: `data23.csv`
**Format**: CSV (Comma-separated values)
**Encoding**: UTF-8
**Rows**: [Number of samples]
**Columns**: [Number of features + outcome]

## Outcome Variable

| Variable | Description | Type | Values | Missing |
|----------|-------------|------|--------|---------|
| `pprom` | Preterm Premature Rupture of Membranes | Binary | 0 = No PPROM, 1 = PPROM | [% missing] |

## Feature Categories

### 1. Demographic Variables

| Variable | Description | Type | Units | Missing |
|----------|-------------|------|-------|---------|
| `age` | Maternal age | Continuous | Years | [% missing] |
| `gestational_age` | Gestational age at admission | Continuous | Weeks | [% missing] |
| `parity` | Number of previous pregnancies | Integer | Count | [% missing] |
| `bmi` | Body Mass Index | Continuous | kg/m² | [% missing] |

### 2. Medical History

| Variable | Description | Type | Values | Missing |
|----------|-------------|------|--------|---------|
| `previous_pprom` | History of previous PPROM | Binary | 0 = No, 1 = Yes | [% missing] |
| `uti_history` | History of urinary tract infections | Binary | 0 = No, 1 = Yes | [% missing] |
| `sti_history` | History of sexually transmitted infections | Binary | 0 = No, 1 = Yes | [% missing] |

### 3. Current Pregnancy

| Variable | Description | Type | Values | Missing |
|----------|-------------|------|--------|---------|
| `multiple_gestation` | Multiple gestation pregnancy | Binary | 0 = No, 1 = Yes | [% missing] |
| `cervical_length` | Cervical length measurement | Continuous | mm | [% missing] |
| `fetal_fibronectin` | Fetal fibronectin test result | Binary | 0 = Negative, 1 = Positive | [% missing] |

### 4. Laboratory Values

| Variable | Description | Type | Units | Normal Range | Missing |
|----------|-------------|------|-------|--------------|---------|
| `wbc` | White blood cell count | Continuous | 10³/μL | 4.5-11.0 | [% missing] |
| `crp` | C-reactive protein | Continuous | mg/L | <10 | [% missing] |
| `il6` | Interleukin-6 | Continuous | pg/mL | <5 | [% missing] |
| `ph_vaginal` | Vaginal pH | Continuous | pH units | 3.8-4.5 | [% missing] |

### 5. Clinical Symptoms

| Variable | Description | Type | Values | Missing |
|----------|-------------|------|--------|---------|
| `vaginal_bleeding` | Presence of vaginal bleeding | Binary | 0 = No, 1 = Yes | [% missing] |
| `uterine_contractions` | Regular uterine contractions | Binary | 0 = No, 1 = Yes | [% missing] |
| `pelvic_pressure` | Pelvic pressure or discomfort | Binary | 0 = No, 1 = Yes | [% missing] |

### 6. Examination Findings

| Variable | Description | Type | Values | Missing |
|----------|-------------|------|--------|---------|
| `cervical_dilation` | Cervical dilation | Ordinal | 0-10 cm | [% missing] |
| `ferning_test` | Fern test result | Binary | 0 = Negative, 1 = Positive | [% missing] |
| `pooling_test` | Pooling test result | Binary | 0 = Negative, 1 = Positive | [% missing] |

## Data Collection

### 1. Time Points
- Baseline assessment at admission
- Follow-up measurements as clinically indicated

### 2. Measurement Methods
- Standardized clinical assessments
- Laboratory analyses per institutional protocols
- Imaging studies with validated techniques

### 3. Quality Control
- Double data entry for key variables
- Range checks for all measurements
- Consistency checks between related variables

## Data Processing

### 1. Cleaning Steps
1. Removal of duplicate records
2. Handling of implausible values
3. Standardization of measurement units
4. Consolidation of categorical variables

### 2. Missing Data
- Percentage missing per variable calculated
- Multiple imputation methods considered
- Sensitivity analyses planned for missing data

### 3. Derived Variables
- Calculation of composite scores
- Creation of interaction terms
- Time-to-event variables if longitudinal data

## Data Security

### 1. Storage
- Encrypted storage devices
- Password-protected files
- Access restricted to study team

### 2. Sharing
- De-identified data only
- Data use agreements required
- Compliance with institutional policies

## Version Control

### 1. Data Versions
- Version 1.0: Initial data collection
- Version 1.1: Data cleaning completed
- Version 1.2: Final analytic dataset

### 2. Changes Log
- [Date]: Added new variables
- [Date]: Corrected data entry errors
- [Date]: Updated outcome definitions

## Contact Information

For questions about the data:
- Principal Investigator: [Name]
- Data Manager: [Name]
- Statistical Analyst: [Name]

Email: [contact@email.com]
Phone: [phone number]
