import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import os

app=Flask(__name__)     # starting point of our app

IMG_FOLDER=os.path.join('static','IMG')
app.config['UPLOAD_FOLDER']=IMG_FOLDER

##Load the model
regmodel=pickle.load(open('Housing_reg_model.pkl', 'rb')) #rb

@app.route('/')
def home():
    return render_template("index.html")       

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]               
    output1=regmodel.predict(np.array(data).reshape(1,-1)).round(3)
   
    return render_template("index.html", prediction_text="The price of House is {}".format(output1[0])
                           ) 
                   
    
if __name__=="__main__":
    app.run(debug=True)