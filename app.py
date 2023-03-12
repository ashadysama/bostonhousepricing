import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

#Load model
model = pickle.load(open('regmodel.pkl', 'rb'))

#Load StandardScaler
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

#Creating predict api
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))

    #scaling the input data
    scaled_data= scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output= model.predict(scaled_data)
    print(output[0])
    return jsonify(output[0])


if __name__== "__main__":
    app.run(debug=True)


