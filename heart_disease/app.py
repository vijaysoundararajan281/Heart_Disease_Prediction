import numpy as np
import pandas as pd
from flask import Flask,request, jsonify, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('class.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_data=[float(x) for x in request.form.values()]
    final_features=[np.array(input_data)]
    prediction=model.predict(final_features)
    output=np.around(prediction)
    if output==1:
        output="Yes"
    elif output==0:
        output="No"
    return render_template('index.html',prediction_text='Heart Disease Prediction: {}'.format(output))
'''
@app.route('/predict_ap',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output=prediction[0]
    return jsonify(output)
'''
if __name__ == "__main__":
    app.run(debug=True)