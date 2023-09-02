#flask, scikit-learn, pandas, pickle-mixin

import pandas as pd
import pickle

from flask import Flask, render_template, request

app = Flask(__name__)
data=pd.read_csv('Cleaned_Housedata.csv')
pipe=pickle.load(open("RidgeHouseModel.pkl",'rb'))

@app.route('/')
def index():

    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():

    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('total_sqft')

    print(location,bhk,bath,sqft)
    input=pd.DataFrame([[location,sqft,bath,bhk]] ,columns=['location','total_sqft','bath','bhk'])
    prediction=pipe.predict(input)[0]


    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
