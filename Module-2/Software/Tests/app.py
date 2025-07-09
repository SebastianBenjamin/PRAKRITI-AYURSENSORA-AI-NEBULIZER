import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model and label encoder
model = joblib.load('models/xgb_prakruthi_model.pkl')
le = joblib.load('models/label_encoder.pkl')

# Define the column order expected by the model (EXACT match with training data)
expected_columns = [
    'Skin color', 'Skin texture', 'Details about hair type', 
    'Texture of hair ', 'My hairs are', 'Hair falling tendency\t', 
    'Eyes\t', 'My eyes  eye balls are', 'Physique\t', 'Veins\t',
    'Usual body temperature \t', 'Teeth \t', 'Is your lips\t',
    'Hands(Length)', 'Body hair\t', 'Joints\t', 'Chest\t', 'Tongue\t',
    'Body frame (Length)\t', 'Body frame (Breadth)\t', 'Bodyweight changes\t',
    'Style of speaking\t', 'Voice \t', 'Walk \t', 'Appetite \t ',
    'Intensity of hunger\t', 'Favorite climate\t', 'Stamina\t',
    'Tastes, I like most', 'Eagerness about drinking water',
    'Frequency of liquid intake\t', 'Sleep\t', 'Quantity of sweat\t',
    'Emotion\t', 'Intellect\t', 'Grasping\t', 'Concentration\t', 'Memory\t',
    'Temperament\t', 'Patience\t', 'Satisfaction/content\t',
    'While making decisions\t', 'Activities\t', 'Nature of dreams\t',
    'Friendships & relations\t', 'Faith in God\t', 'Self-control\t',
    'Gratefulness\t', 'Likings\t', 'Liking- type of food',
    'Politeness/humbleness', 'Biting nails ', 'Fond of praise ', 'Brave', 'Egoistic'
]

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Dosha Prediction API',
        'instructions': 'Send a POST request to /predict with feature values to get dosha prediction'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json(force=True)
        
        # Create a DataFrame with the expected columns
        input_df = pd.DataFrame(columns=expected_columns)
        
        # Fill the DataFrame with the input data
        for col in expected_columns:
            if col in data:
                # Handle null/empty values
                if data[col] is None or str(data[col]).strip() == '':
                    return jsonify({'error': f'Missing value for feature: {col}'}), 400
                
                # Convert to integer with validation
                try:
                    value = int(float(data[col]))
                    if value not in [1, 2, 3]:
                        return jsonify({'error': f'Value for {col} must be 1, 2, or 3'}), 400
                    input_df.at[0, col] = value
                except (ValueError, TypeError):
                    return jsonify({'error': f'Invalid value for {col}. Must be numeric (1, 2, or 3).'}), 400
            else:
                return jsonify({'error': f'Missing feature: {col}'}), 400
        
        # Ensure all columns have integer data types
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            if input_df[col].isna().any():
                return jsonify({'error': f'Invalid value for {col}. Could not convert to number.'}), 400
        
        # Make prediction
        prediction = model.predict(input_df)
        predicted_class = le.inverse_transform(prediction)
        
        # Get probability scores
        probabilities = model.predict_proba(input_df)[0]
        class_probabilities = {le.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(probabilities)}
        
        return jsonify({
            'prediction': predicted_class[0],
            'probabilities': class_probabilities
        })
        
    except ValueError as ve:
        return jsonify({'error': f'Value error: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)