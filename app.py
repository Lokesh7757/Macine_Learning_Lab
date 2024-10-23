import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
# Load dataset
url = 'https://drive.google.com/uc?id=1XCv1c5Qero8-7Sg0xRc2Tqy_4Y0iEgnQ'
data = pd.read_csv(url)

# Display the first few rows
st.write(data.head())

# Handle missing data
data.fillna(data.mode().iloc[0], inplace=True)  # Filling missing values with mode for categorical features

# Convert categorical variables to numerical
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Split the data into features and target variable
X = data.drop('lung_cancer', axis=1)  # Replace 'lung_cancer' with the actual target column name
y = data['lung_cancer']  # Assuming 'lung_cancer' is the target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
report = classification_report(y_test, y_pred)

st.write(f'Accuracy: {accuracy:.2f}')
st.write(f'ROC-AUC Score: {roc_auc:.2f}')
st.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(plt)

# Feature importance
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

st.write('Feature Importance:')
st.bar_chart(feature_importance_df.set_index('Feature'))

