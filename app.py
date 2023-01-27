import pickle
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the pickle file
with open('modelNB.pkl', 'rb') as file:
    model = pickle.load(file)

# Import Flask for creating API
from flask import Flask, request

# Initialise a Flask object
app = Flask(__name__)

# Create an API endpoint for predicting

@app.route('/predict')
def predict_cancer():
    # Read all necessary request parameters
    income = request.args.get('income')
    age = request.args.get('age')
    loan = request.args.get('loan')

    #prediction for unseen data
    unseen_data = np.array([[income, age, loan]]).astype(np.float64)

    prediction  = model.predict(unseen_data)
    output = prediction[0]

    # return the result
    reponse = {'Loan Status' : 'Approved' if output == 0 else 'Declined'}

    return {
        'statusCode': 200,
        'body': json.dumps(reponse)
    }

#Run Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
