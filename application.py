import pickle
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the pickle file
with open('modelNB.pkl', 'rb') as file:
    model = pickle.load(file)

# Import Flask for creating API
from flask import Flask, request
from flask_cors import CORS

# Initialise a Flask object
application = Flask(__name__)
CORS(application)

# Create an API endpoint for predicting

@application.route('/predict')
def predict_cancer():
    # Read all necessary request parameters
    income = request.args.get('income')
    age = request.args.get('age')
    loan = request.args.get('loan')

    #prediction for unseen data
    unseen_data = np.array([[income, age, loan]]).astype(np.float64)

    x = unseen_data
    print(x)
    prediction = model.predict(x)
    print(prediction[0])

    output = prediction[0]

    # return the result
    reponse = {'Loan Status' : 'Approved' if output == 0 else 'Declined'}

    return reponse
    
#Run Server
if __name__ == '__main__':
    application.run()
