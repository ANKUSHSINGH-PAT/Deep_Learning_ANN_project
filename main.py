from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and preprocessing tools
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

try:
    with open('LabelEncoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('geo_encoder.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scalar.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Error loading preprocessing tools: {e}")

# Define input data model
class PredictionInput(BaseModel):
    geography: str
    gender: str
    age: int
    balance: float
    credit_score: float
    estimated_salary: float
    tenure: int
    num_of_products: int
    has_cr_card: int
    is_active_member: int

@app.get('/')
def home():
    return{"hello"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Prepare the input data
    data_dict = input_data.dict()
    gender_encoded = label_encoder_gender.transform([data_dict['gender']])[0]
    geo_encoded = onehot_encoder_geo.transform([[data_dict['geography']]]).toarray()

    # Convert to DataFrame
    input_data_df = pd.DataFrame([{
        'CreditScore': data_dict['credit_score'],
        'Gender': gender_encoded,
        'Age': data_dict['age'],
        'Tenure': data_dict['tenure'],
        'Balance': data_dict['balance'],
        'NumOfProducts': data_dict['num_of_products'],
        'HasCrCard': data_dict['has_cr_card'],
        'IsActiveMember': data_dict['is_active_member'],
        'EstimatedSalary': data_dict['estimated_salary']
    }])
    
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data_df = pd.concat([input_data_df.reset_index(drop=True), geo_encoded_df], axis=1)

    # Ensure that the input_data columns are in the correct order
    expected_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'] + list(geo_encoded_df.columns)
    input_data_df = input_data_df[expected_columns]

    # Scale the input data
    try:
        input_data_scaled = scaler.transform(input_data_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scaling input data: {e}")

    # Predict churn
    try:
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

    # Return the result
    return {
        'churn_probability': round(prediction_proba, 2),
        'likely_to_churn': prediction_proba > 0.5
    }

# Run the app with `uvicorn filename:app --reload`
if __name__=="__main__":
    uvicorn.run(app,host="127.0.0.1",port=2000)