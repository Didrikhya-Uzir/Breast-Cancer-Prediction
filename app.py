# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:59:11 2023

@author: didri
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
import os
model_path = 'classifier.keras'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    print(f"The file '{model_path}' does not exist.")
    print(f"Current working directory: {os.getcwd()}")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    val_features = [float(x) for x in request.form.values()]
    final_features = np.array(val_features).reshape(1, -1)
    prediction_prob = model.predict(final_features)[0]
    
    # Choose the class with the highest probability
    predicted_class = np.argmax(prediction_prob)
    print(predicted_class)
    
    if predicted_class == 1:
        output = "The tumor is Benign"
    else:
        output = "The tumor is Malignant"
    
    return render_template("index.html", prediction_text=output)



if __name__ == "__main__":
    app.run(debug=True)
    