import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

application = Flask(__name__) #Initialize the flask App
model = pickle.load(open('student_tree.pkl', 'rb'))

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
    
    if output == "[2]":
        hasil = "Cluster 3"
    elif output == "[1]":
        hasil = "Cluster 2"
    elif output == "[3]":
        hasil = "Cluster 4"
    elif output == "[4]":
        hasil = "Cluster 5"
    else:
        hasil = "Cluster 1"

    return render_template('index.html', prediction_text= hasil)

if __name__ == "__main__":
    application.run(debug=True)
