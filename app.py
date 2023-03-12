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

#Prediction from web page
@app.route('/predict', methods=['POST'])
def predict():
    data= [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output= model.predict(final_input)[0]
    return render_template("home.html", prediction_text= "The predicted house price (in 1000's $) is: {}".format(output))



if __name__== "__main__":
    app.run(debug=True)


