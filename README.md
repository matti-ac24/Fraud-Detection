# Credit Card Fraud Detection

This project focuses on detecting fraudulent transactions using machine learning models. The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains anonymized features and a highly imbalanced class distribution.

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
