# Seed-Assisted Zeolite Synthesis Prediction Project

## Project Overview
This project focuses on developing machine learning models to predict the success of seed-assisted zeolite synthesis experiments. The goal is to create a predictive model that can help determine whether a synthesis experiment will result in a pure zeolite phase (success) or undesired phases (failure).

## Dataset Description
The dataset contains 385 historical records of seed-assisted zeolite synthesis experiments with the following features:

### Input Features
1. Seed Properties:
   - Seed type
   - Seed amount (normalized to SiO2 weight = 1)
   - Seed framework density (FD) in T/Å3
   - Seed Si/Al molar ratio (measured using ICP-AES)

2. Gel Composition:
   - SiO2 (normalized to 1)
   - NaOH/SiO2 molar ratio
   - B2O3/SiO2 molar ratio
   - H2O/SiO2 molar ratio
   - OTMAC/SiO2 molar ratio (SDA)

3. Crystallization Conditions:
   - Crystallization temperature (°C)
   - Crystallization time (days)

### Target Variable
- Class "0": Failed experiments (amorphous, mixed, dense, or layered phases)
- Class "1": Successful experiments (pure zeolite phase)

## Analysis Steps

1. Data Preprocessing:
   - Handling outliers
   - Feature scaling (StandardScaler for numerical features)
   - One-hot encoding for categorical variables

2. Correlation Analysis:
   - Created correlation heatmaps
   - Analyzed feature relationships with the target variable

3. Model Development:
   - Implemented multiple classification models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Neural Network

4. Model Evaluation:
   - Performance metrics used:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - AUC-ROC

5. Feature Importance Analysis:
   - Feature importance plots for tree-based models
   - Partial Dependence Plots (PDP) for understanding feature impacts

## Files Description
- `data_preprocessing.py`: Code for data cleaning and preprocessing
- `model_training.py`: Implementation of machine learning models
- `evaluation.py`: Model evaluation and visualization code
- `requirements.txt`: List of required Python packages

## Results
(Specific model performances and key findings can be added here)

## Requirements
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Usage
```python
# Example code for running the analysis
python model_training.py