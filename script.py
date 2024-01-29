# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load your dataset (replace 'credit_card_transactions.csv' with your actual dataset)
data = pd.read_csv('credit_card_transactions.csv')

# 1. Data Cleaning
# Assuming 'fraud_label' is the target variable
X = data.drop('fraud_label', axis=1)
y = data['fraud_label']

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Handle outliers (you may need to customize based on your data)
lower_threshold = X.quantile(0.05)
upper_threshold = X.quantile(0.95)
X = X[(X > lower_threshold) & (X < upper_threshold)]

# Handle multi-collinearity (you may need to customize based on your data)
correlation_matrix = X.corr().abs()
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
X = X.drop(to_drop, axis=1)

# 2. Fraud Detection Model
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 3. Variable Selection
# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
selected_features = feature_importances[feature_importances > 0.02].index

X_selected = X[selected_features]

# 4. Model Performance
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# 5. Key Factors
# Display feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
key_factors = feature_importances.nlargest(5).index

print("Key Factors:")
print(key_factors)

# 6. Do these factors make sense? (Provide insights based on your analysis)
# In this case, you might find that transaction amount, frequency, and location are key factors.

# 7. Prevention Measures (Provide insights based on your analysis)
# Implementing two-factor authentication, monitoring transactions in high-risk locations, etc.

# 8. Implementation Verification (Hypothetical code)
# Assume you have a new dataset after infrastructure update
new_data = pd.read_csv('new_credit_card_data_after_update.csv')  # Replace with actual dataset path
new_X = new_data[selected_features]

# Make predictions on the new data using the trained model
new_predictions = model.predict(new_X)

# Assuming a decrease in fraud cases indicates success
if new_predictions.sum() < y.sum():
    print("Prevention measures seem effective.")
else:
    print("Further analysis is needed to assess the impact of prevention measures.")
