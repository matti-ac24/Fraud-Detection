# Credit Card Fraud Detection Using Machine Learning

This project aims to build a machine learning pipeline for detecting fraudulent transactions using the highly imbalanced [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset includes credit card transactions made by European cardholders in September 2013, where fraudulent transactions represent only 0.172% of all transactions.

---

## Project Overview

Fraud detection is critical in the financial sector, requiring effective handling of highly imbalanced datasets. This project employs **Logistic Regression** and **Random Forest Classifiers** to address this challenge, alongside techniques for data preprocessing, imbalance handling, and hyperparameter optimization.

### Key Highlights:

- **Data Preprocessing**:
  - Normalized the "Amount" feature using `StandardScaler`.
  - Dropped irrelevant features such as "Time" to focus on meaningful predictors.

- **Handling Imbalanced Data**:
  - Used **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples for the minority class (fraudulent transactions).
  - Balanced data improves the model's ability to detect fraud without biasing toward the majority class.

- **Model Selection**:
  - **Logistic Regression**: Chosen as a simple, interpretable baseline for binary classification.
  - **Random Forest**: Selected for its robustness, ability to capture complex patterns, and feature importance analysis.

- **Hyperparameter Tuning**:
  - Applied **RandomizedSearchCV** for Random Forest, which is faster than GridSearchCV and finds near-optimal hyperparameters.

- **Evaluation Metrics**:
  - **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)**: Measures overall model performance.
  - **AUPRC (Area Under the Precision-Recall Curve)**: Focuses on the minority class, offering a better evaluation for imbalanced datasets.

---

## Dependencies

The project requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

---

## Project Structure

- **Dataset**: The dataset was sourced from Kaggle ([link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) and is not included in this repository due to size limitations.
- **Code**: Contains data preprocessing, model training, hyperparameter tuning, and evaluation.
- **Visualizations**: Includes dynamically generated plots for performance metrics and feature importance.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Fraud-Detection.git
   cd Fraud-Detection

---

## **Dataset**

The dataset contains transactions made by European cardholders in September 2013. It is highly imbalanced, with fraudulent transactions accounting for only 0.172% of all transactions. Due to its size (over 100 MB), the dataset is not included in this repository. 

To use the dataset:
1. Download it from [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the `creditcard.csv` file in the `Dataset/` directory.

---

## **Machine Learning Models**

### **1. Logistic Regression**
- **Why Logistic Regression?**
  Logistic Regression is a simple and interpretable linear model that works well for binary classification problems. In this project, it serves as a baseline model to compare with more complex algorithms.
- **Key Feature:**
  The `class_weight='balanced'` parameter adjusts for the imbalanced dataset by assigning higher weights to the minority class (fraudulent transactions).

### **2. Random Forest**
- **Why Random Forest?**
  Random Forest is an ensemble learning method that uses multiple decision trees to achieve better performance and reduce overfitting. It is robust to noise and works well with imbalanced datasets.
- **Tuning with RandomizedSearchCV:**
  - **What is RandomizedSearchCV?**
    RandomizedSearchCV is a hyperparameter optimization technique that selects random combinations of parameters for testing. It is faster than GridSearchCV, especially for large datasets and models with many hyperparameters.
  - **Why RandomizedSearchCV?**
    It provides a good balance between efficiency and performance improvement, allowing us to tune the Random Forest model effectively.

---

## **Key Techniques**

### **1. SMOTE (Synthetic Minority Oversampling Technique)**
- **Why SMOTE?**
  The dataset is highly imbalanced, with only 0.172% of transactions being fraudulent. SMOTE generates synthetic samples for the minority class, balancing the dataset and improving model performance.
- **Implementation:**
  Applied only to the training set to prevent data leakage.

### **2. Evaluation Metrics**
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
  - Measures the model's ability to distinguish between classes.
  - Higher ROC-AUC indicates better model performance.
- **AUPRC (Area Under the Precision-Recall Curve):**
  - Particularly useful for imbalanced datasets.
  - Measures the trade-off between precision and recall for the positive class.

---

## **Results**

### **1. Logistic Regression**
- **ROC-AUC:** 0.9700
- **AUPRC:** 0.7213

### **2. Random Forest (Tuned)**
- **ROC-AUC:** 0.9691
- **AUPRC:** 0.8756

---

## **How to Run the Code**

1. Clone this repository:
   ```bash
   git clone https://github.com/matti-ac24/Fraud-Detection.git
