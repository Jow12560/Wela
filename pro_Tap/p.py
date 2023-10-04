import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix,precision_recall_curve, average_precision_score
import pickle
import joblib
import matplotlib as mpl


db = pd.read_csv('cen_eat.csv')


# Keep type
type = []
# Iterate through the 'business_category' column and add unique values to the list
for business_type in db['business_type']:
    if business_type not in type:
        type.append(business_type)



# Extract features and target variable
X = db[['business_type', 'profit_margin']].copy()
y = db['district'].copy()

# Encode categorical features using One-Hot Encoding
X = pd.get_dummies(X, columns=['business_type'], prefix=['business_type'])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# Standardize numerical features
scaler = StandardScaler()
X_train[['profit_margin']] = scaler.fit_transform(X_train[['profit_margin']])
X_test[['profit_margin']] = scaler.transform(X_test[['profit_margin']])





# Create and train a Logistic Regression classifier
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with Logistic Regression: {accuracy * 100:.2f}%")



# Define the values for 'business_type' and 'profit_margin' for your test case
new_business_type = 'โรงแรม รีสอร์ทและห้องชุด'  # Replace with the actual business type
new_profit_margin = -2959.524028  # Replace with the actual profit margin value

# Create a new DataFrame with the input data for testing
new_data = pd.DataFrame({'business_type': [new_business_type], 'profit_margin': [new_profit_margin]})

# Ensure that new_data has the same columns as X_train with the correct order
new_data = new_data.reindex(columns=X_train.columns, fill_value=False)

# Standardize the 'profit_margin' column in new_data using the same scaler
new_data[['profit_margin']] = scaler.transform(new_data[['profit_margin']])

# Make predictions with the model
predicted_district = clf.predict(new_data)

print(f"Predicted district: {predicted_district[0]}")



# filename = 'ml.pkl'
# # Serialize the model
# pickle.dump(clf, open(filename, 'wb'))
# # Serialize the ColumnTransformer
# joblib.dump(scaler, 'scaler.pkl')






















