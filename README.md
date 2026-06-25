# Airline Passenger Satisfaction Prediction

## Description
This Machine Learning project aims to predict airline passenger satisfaction based on flight characteristics and passenger demographic information.

The problem is formulated as a supervised binary classification task (Satisfied / Dissatisfied).

The project includes the full ML pipeline: data preprocessing, statistical analysis, feature selection, model training, hyperparameter tuning, performance evaluation, and efficiency analysis.

---

## Objective
The goal is to build a predictive model capable of classifying passenger satisfaction using features such as:
- flight-related attributes (delays, service quality, comfort, travel type, etc.)
- demographic information (age, gender, customer type, etc.)

---

## Dataset
The project uses two datasets:
- `train.csv` → used for training and validation
- `test.csv` → used for final testing

---

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Feature Selection & Statistical Analysis
Feature selection and statistical analysis were performed to better understand relationships between variables and improve model performance.

- StandardScaler for feature normalization
- Chi-Square test (chi2) for categorical feature relevance
- Mutual Information (mutual_info_classif) for non-linear feature importance
- Independent t-test (ttest_ind) to compare feature distributions between groups and identify statistically significant differences
- Scikit-learn Pipeline for structured preprocessing and model building

---

## Models Used
- DummyClassifier (baseline model)
- Logistic Regression
- Random Forest Classifier
- AdaBoost Classifier

---

## Model Selection & Optimization
- Cross-validation (cross_val_score)
- Hyperparameter tuning using GridSearchCV
- Fixed random seed for reproducibility (SEED = 48)

Training times were measured during hyperparameter tuning to evaluate computational cost across model configurations.

---

## Evaluation Metrics

The models were evaluated using:

### Accuracy
\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

### Precision
\[
Precision = \frac{TP}{TP + FP}
\]

### Recall
\[
Recall = \frac{TP}{TP + FN}
\]

### F1-Score
\[
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
\]

### ROC-AUC
Measures the model’s ability to distinguish between classes across different thresholds.

Additional evaluation tools:
- Classification Report
- Confusion Matrix
- ROC Curve

---

## Performance Analysis

### Training Time (Hyperparameter Optimization)
Training time was measured during GridSearchCV to evaluate the computational cost of hyperparameter tuning, including cross-validation across multiple parameter combinations.

---

### Inference Time
Inference time was measured to evaluate model efficiency in real-world usage, focusing on:
- prediction speed on unseen data
- scalability of each model
- trade-off between accuracy and computational cost

---

### Feature Importance (Random Forest)
A feature importance analysis was performed using the Random Forest model to identify the most relevant variables influencing passenger satisfaction.

Key influencing factors include:
- service quality indicators
- travel class and type
- in-flight comfort features
- delay-related variables

---

### Overfitting Analysis
An overfitting analysis was conducted by comparing training performance with cross-validation performance.

This includes:
- comparison of training vs validation scores
- evaluation of generalization capability
- detection of overfitting in complex models (e.g., Random Forest, AdaBoost)

This analysis helped select models with the best balance between accuracy and generalization.

---

## Repository Structure
- `Airline_Passenger_Satisfaction.ipynb` → full analysis, preprocessing, feature selection, and model training
- `train.csv` → training dataset
- `test.csv` → test dataset
- `requirements.txt` → project dependencies
- `Airline Passenger Satifaction.pdf` → project documentation

---

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/MatteoMaioli/Airline-Passenger-Satisfaction.git

```

2.Install dependencies:

pip install -r requirements.txt

3. Open the notebook

Open Airline_Passenger_Satisfaction.ipynb using:

Jupyter Notebook
VS Code
Google Colab

4. Run all cells

Execute the notebook from top to bottom to reproduce the analysis.

---

## Note

This project was developed for academic purposes to apply supervised machine learning techniques to a real-world classification problem, including statistical analysis, feature selection, and performance evaluation.
