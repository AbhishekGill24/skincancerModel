#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
# from main import getPrediction
import os
from pyngrok import ngrok
import uvicorn
from flask_ngrok import run_with_ngrok

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


# In[2]:


def getPrediction(filename):
    
    classes = ['Actinic keratoses', 'Basal cell carcinoma', 
               'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
               'Melanocytic nevi', 'Vascular lesions']
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    
    
    #Load model
    my_model=load_model("models/model.h5")
    
    SIZE = 32 #Resize to same size as training images
    img_path = 'static/images/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = my_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Diagnosis is:", pred_class)
    return pred_class


# In[4]:


UPLOAD_FOLDER = 'static/images/'
app = Flask(__name__, static_folder="static")
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)  
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            #getPrediction(filename)
            label = getPrediction(filename)
            flash(label)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash(full_filename)


if __name__ == '__main__':
   app.run(port=5002)


# In[ ]:




