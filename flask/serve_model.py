from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

# Load model
model_path = r'C:\Users\ftesfaye\Desktop\KIFIYA\KIFIYA_PROJECT_WEEK_8\Improved-Fraud-Detection-for-E-commerce-and-Bank-Transactions\data\pickle\DecisionTreeClassifier().pkl'  # Adjust path if necessary
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Endpoint for fraud prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    logger.info(f"Received request: {data}, Prediction: {prediction}")
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
