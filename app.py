from flask import Flask, json, request, jsonify
import numpy as np

import sys
import os
import glob
import re
import numpy as np
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='modelV8.h5'

# Load your trained model
model = load_model(MODEL_PATH)



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

   

    preds = model.predict(x)
    # preds=np.argmax(preds, axis=1)
    # if preds==0:
    #     preds="Hiace"
    # elif preds==1:
    #     preds=" Noah"
    # elif preds== 2:
    #     preds='Premio'
    # else:
    #     preds="Skyline"
    
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds = {
            "make" : 'Toyota',
            "model" : 'Allion',
            "bodyType" : "Sedan"
        }
    elif preds==1:
        preds = {
            'make' : 'Nissan',
            'model' : 'Caravan',
            "bodyType" : "Van"
        }
        
    elif preds== 2:
        preds = {
            'make' : 'Toyota',
            'model' : 'Hiace',
            "bodyType" : "Van"
        }
    elif preds == 3: 
            preds = {
                'make' : 'Toyota',
                'model' : 'Premio',
                "bodyType" : "Sedan"
            }
    else: 
        
        preds = {
            'make' : 'Honda',
            'model' : 'Vezel',
            "bodyType" : "SUV"
        }
    
    
    return json.dumps(preds)


@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    name = request.args['name']
    print(name)
    return name



@app.route('/name', methods=['GET', 'POST'])
def view():
    data = request.json
    print(data['name'])
    return data['name']



@app.route('/image', methods=['GET', 'POST'])
def predict():
    f = request.files['image']

      # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
    basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)


    preds = model_predict(file_path, model)
    result=preds
    return result
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)