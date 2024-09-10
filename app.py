# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from asd_inputs import asd_input
import numpy as np
import pickle
import pandas as pd


# 2. Create the app object
app = FastAPI()
pickle_in = open("lda_model.pkl","rb")
classifier=pickle.load(pickle_in)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_autism(data:asd_input):
    df_loaded = pd.read_csv('asd_data.csv')
    test_x = pd.DataFrame([data.dict()])
    test_x_encoded = pd.get_dummies(test_x, columns=test_x.columns)
    df_loaded, test_x_encoded = df_loaded.align(test_x_encoded, join='left', axis=1)
    test_x_encoded = test_x_encoded.fillna(0)
    input_data_normalized = scaler.transform(test_x_encoded)
    prediction = classifier.predict(input_data_normalized)
    if(prediction[0]==0):
        prediction="No Autism"
    else:
        prediction="There is a chance for ASD"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
#if __name__ == '__main__':
    #uvicorn.run(app, host='0.0.0.0', port=10000)
    
#uvicorn app:app --reload
