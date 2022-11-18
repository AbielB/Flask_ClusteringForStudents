import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

application = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model_tree.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = format(prediction)
    if output == "[0.]":
        hasil = "Normal"
    else:
        hasil = "Ada Penyakit"

    return render_template('index.html', prediction_text= hasil)

if __name__ == "__main__":
    application.run(debug=True)
