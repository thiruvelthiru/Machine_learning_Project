import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)[0][0]
    return render_template('index.html', prediction_text='Sales Prediction is {}'.format(round(prediction,2)))


@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = json.loads(request.data)
    TV_Value=data['TV_Value']
    prediction = model.predict([[TV_Value]])[0][0]
    return json.dumps({'Sales Prediction': round(prediction,2) })

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)