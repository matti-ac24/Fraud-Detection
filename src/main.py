import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# Create a directory for saving plots
if not os.path.exists("Plots"):
    os.makedirs("Plots")

# Load the Dataset
df = pd.read_csv("Dataset/creditcard.csv")

# Inspect the dataset for null values and class distribution
print(df.info())
print("\n ------------------------------------ \n")
print(df['Class'].value_counts())

# Separate the dependent features from the target feature
X = df.drop(columns=["Class", "Time"])  # Drop 'Time' as it's irrelevant for our model
Y = df["Class"]

# Normalize the 'Amount' column
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# Split into train and test sets (stratify ensures the same class ratio in both sets)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Print the class distribution after SMOTE
print("Class distribution before SMOTE:", Y_train.value_counts())
print("Class distribution after SMOTE:", pd.Series(Y_train_resampled).value_counts())

# Train Logistic Regression
logistic_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
logistic_model.fit(X_train_resampled, Y_train_resampled)

# Hyperparameter Tuning for Random Forest
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=10,  # Reduced iterations for faster execution
    scoring='average_precision',  # AUPRC as the evaluation metric
    cv=3,  # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available processors
)

print("Starting RandomizedSearchCV for Random Forest...")
rf_random_search.fit(X_train_resampled, Y_train_resampled)

# Print the best parameters and best score
print("\nBest Parameters from RandomizedSearchCV:")
print(rf_random_search.best_params_)
print(f"Best AUPRC: {rf_random_search.best_score_:.4f}")

# Best Random Forest Model
best_rf_model = rf_random_search.best_estimator_

# Feature Importance Analysis for Random Forest
if hasattr(best_rf_model, "feature_importances_"):
    feature_importances = best_rf_model.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    # Plot Feature Importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("Plots/Feature_Importance_Random_Forest.png")
    plt.show()

    print("\nTop 5 Most Important Features:")
    print(importance_df.head())

# Evaluate Models
models = {
    "Logistic Regression": logistic_model,
    "Random Forest (Tuned)": best_rf_model
}

plt.figure(figsize=(10, 8)) # create new plot for combined PR curve

for model_name, model in models.items():
    # Predictions
    Y_pred = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)[:, 1]  # Predicted probabilities for the positive class

    # Metrics
    roc_auc = roc_auc_score(Y_test, Y_proba)
    precision, recall, _ = precision_recall_curve(Y_test, Y_proba)
    auprc = average_precision_score(Y_test, Y_proba)

    print(f"\n{model_name}")
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    # Plot Precision-Recall Curve
    plt.plot(recall, precision, label=f"{model_name} (AUPRC = {auprc:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: Logistic Regression & Random Forest")
plt.legend()
plt.grid()

# Delete the old single Logistic Regression plot if it exists
logistic_curve_path = "Plots/Logistic_Regression_PR_Curve.png"
if os.path.exists(logistic_curve_path):
    os.remove(logistic_curve_path)

# Save only the final combined plot
plt.savefig("Plots/Logistic_Regression_Random_Forest_PR_Curve.png")
plt.show()