import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


app = Flask(__name__)
from tensorflow.keras.models import load_model
import os
model=load_model(os.path.join(os.path.abspath(os.path.dirname(__file__)),"insta.h5"))


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    if(int_features[5]=='yes'):
        int_features[5] = 1
    else:
        int_features[5] = 0
    if (int_features[6] == 'yes'):
        int_features[6] = 1
    else:
        int_features[6] = 0
    if (int_features[7] == 'yes'):
        int_features[7] = 1
    else:
        int_features[7] = 0

    int_features_new = [int(x) for x in int_features]
    final_features = [np.array(int_features_new)]
    u = pd.DataFrame(final_features, columns=['profile pic', 'fullname words', 'name==username', 'description length', 'private',
                                 '#posts', '#followers', '#follow'])

    prediction = model.predict_classes(u)

    if(prediction[0]==1):
        output = "Yes"
    else:
        output = "No"
    return render_template('index.html', prediction_text= 'Fake: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)