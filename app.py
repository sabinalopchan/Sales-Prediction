from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessing objects
lr_model = joblib.load('linear_regression_model.pkl')
# ohe = joblib.load('one_hot_encoder.pkl')
# scaler = joblib.load('scaler.pkl')

# Define the route for the home page


@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form.to_dict()

    # Convert the data to a DataFrame
    input_data = pd.DataFrame([data])

    # One-hot encode the categorical variables
    input_data_ohe = ohe.transform(input_data[category_columns])

    # Scale the numerical variables
    input_data_scaled = scaler.transform(input_data[numerical_columns])

    # Combine the preprocessed categorical and numerical data
    input_data_preprocessed = np.hstack((input_data_ohe, input_data_scaled))

    # Make the prediction
    prediction = lr_model.predict(input_data_preprocessed)

    return render_template('index.html', prediction_text=f'Predicted Sales: ${prediction[0]:.2f}')


if __name__ == '__main__':
    app.run(debug=True)
