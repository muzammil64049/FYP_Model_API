# Importing Modules
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
# Import label encoder
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import heapq
import pickle
from flask import Flask, request,json,jsonify

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))
LabelEncoder = pickle.load(open("LabelEncoder.pkl", "rb"))

# Create a Flask app
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Parse the input data from the request
    # data = json.loads(request.data)
    data = request.json
    
    # print(data)
    input_data1 = int(data['budget'])
    input_data2 = int(data['People'])
    input_data3 = data['Cuisines']
   # label_encoder = preprocessing.LabelEncoder()
    Cuisine = LabelEncoder.transform([input_data3])
    # Use the model to make predictions
    lsttt = model.predict_proba([[input_data1, input_data2, Cuisine]])

    myRes = ['Asia Live',
             ' Bar B.Q Tonight ',
             'Bhookemoo',
             ' Burger Lab',
             'Burger O Clock ',
             'CO2 The Soda Shop',
             ' Cafe Bistrovia ',
             ' Cafe Filete',
             ' Cafe Piyala ',
             'Cafe Qabail ',
             ' Chaupal Buffet ',
             'Clock Tower - The Food Bazaar',
             'Coffee Wagera',
             'E Street Mews Café',
             ' FLOC - for the love of coffee',
             'Frenchies',
             'Gloria Jeans ',
             'Great Wall ',
             ' Karachi Broast',
             'Kebabjees',
             ' LalQila Restaurant Karachi',
             ' Luqmaah Spicy',
             'Nawabish',
             ' Oh My Grill',
             ' Pizza Max',
             'Qabeela Restaurant',
             ' Rangoli Restaurant',
             'Rosati Bistro',
             'SKY BBQ ',
             ' Seafront - BBQ & Grill',
             'Studio 7teas',
             'Tasty Foods ',
             'The Chefs Cafe Official',
             'The Sauce Burger Cafe',
             'The Soda shop',
             'United King',
             'Zameer Ansari - Gulshan-e-Iqbal Chapter']
    bestR = heapq.nlargest(3, lsttt[0])
    lstN = list(lsttt[0])
    tmp1 = lstN.index(bestR[0])
    lstN[tmp1] = 'NaN'
    tmp2 = lstN.index(bestR[1])
    lstN[tmp2] = 'NaN'
    tmp3 = lstN.index(bestR[2])
    lstN[tmp3] = 'NaN'

    # Return the predictions as a response
    # return {"predictions 1": myRes[tmp1],
    #         "predictions 2:": myRes[tmp2],
    #         "predictions 3:": myRes[tmp3]
    #         }
    response = {
      "isSuccess": True,
      "message": "Successful",
     "predictions_1": myRes[tmp1],
            "predictions_2:": myRes[tmp2],
            "predictions_3:": myRes[tmp3]
      
    }
    # print(response)
    return  jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
