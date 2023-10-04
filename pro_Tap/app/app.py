from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
db = pd.read_csv('cen_eat.csv')


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


# Load the trained model and scaler
model = joblib.load('ml.pkl')
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()


    # Define the values for 'business_type' and 'profit_margin' for your test case
    new_business_type = 'โรงแรม รีสอร์ทและห้องชุด'  # Replace with the actual business type
    new_profit_margin = -2959.524028  # Replace with the actual profit margin value

    # Create a new DataFrame with the input data for testing
    new_data = pd.DataFrame({'business_type': [new_business_type], 'profit_margin': [new_profit_margin]})

    # Ensure that new_data has the same columns as X_train with the correct order
    new_data = new_data.reindex(columns=X_train.columns, fill_value=False)

    # Standardize the 'profit_margin' column in new_data using the same scaler
    new_data[['profit_margin']] = scaler.transform(new_data[['profit_margin']])
    predictions = model.predict(new_data)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(debug=True)


