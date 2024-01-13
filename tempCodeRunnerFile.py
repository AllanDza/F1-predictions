
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

driver_dict = pickle.load(open('driver_dict', 'rb'))
constructor_dict = pickle.load(open('consructor_dict', 'rb'))
model = pickle.load(open('trial.pkl', 'rb'))
data = pd.read_csv('cleaned_data.csv')

trialapp = Flask(_name_)

y_dict = {1: 'Podium Finish',
          2: 'Points Finish',
          3: 'No Points Finish'}

le_d = LabelEncoder()
le_c = LabelEncoder()
le_gp = LabelEncoder()

# Assuming 'driver', 'constructor', 'GP_name' are categorical columns in your dataset
y = le_d.fit_transform(data['driver']).ravel()
le_c.fit_transform(data['constructor']).ravel()
le_gp.fit_transform(data['GP_name']).ravel()

def pred(driver, constructor, quali, circuit):
    gp = le_gp.transform([circuit]).max()
    quali_pos = int(quali)  # Convert quali to float or the appropriate numeric type
    constructor_enc = le_c.transform([constructor]).max()
    driver_enc = le_d.transform([driver]).max()
    driver_confidence = driver_dict[driver].max()
    constructor_reliability = constructor_dict[constructor].max()
    prediction = model.predict([[gp, quali_pos, constructor_enc, driver_enc]]).max()
    return prediction

@trialapp.route('/')
def get_input():
    return render_template('index.html')

@trialapp.route('/predict', methods=['POST', 'GET'])
def predict():
    driver = request.form['driver']
    constructor = request.form['constructor']
    quali = request.form['grid']
    circuit = request.form['circuit']
    my_prediction = pred(driver, constructor, quali, circuit)
    my_prediction = y_dict[my_prediction]
    return render_template('index.html', prediction=my_prediction)

if _name_ == '_main_':
    trialapp.run(debug=False)