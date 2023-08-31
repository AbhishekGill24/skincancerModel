import os
import numpy as np
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Create a Flask web application
app = Flask(__name__, static_folder="static")

# Set a secret key for the application (used for session management)
app.secret_key = "secret"

# Define the folder where uploaded images will be stored
app.config['UPLOAD_FOLDER'] = 'static/images/'

# Function to classify skin conditions using a trained model
def classify_skin_condition(filename):
    labels = ['Actinic keratoses', 'Basal cell carcinoma',
              'Benign keratosis-like lesions', 'Dermatofibroma',
              'Melanoma', 'Melanocytic nevi', 'Vascular lesions'] 
    
    # Create a label encoder and fit it to the labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Load a pre-trained Keras model for classification
    model = load_model("models/model.h5")
    
    image_size = 32
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Open and resize the image using PIL (Python Imaging Library)
    image = np.asarray(Image.open(image_path).resize((image_size, image_size)))
    
    # Normalize pixel values to the range [0, 1]
    image = image / 255.0
    
    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)
    
    # Perform predictions using the loaded model
    predictions = model.predict(image)
    
    # Inverse transform the predicted label index to get the actual class label
    predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])[0]
    
    return predicted_label

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling file uploads and skin condition classification
@app.route('/', methods=['POST'])
def upload_file():
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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Classify the uploaded image and get the diagnosis
            diagnosis = classify_skin_condition(filename)
            
            # Flash the diagnosis and file path to display in the web interface
            flash(diagnosis)
            flash(file_path)

# Run the application if this script is executed directly
if __name__ == '__main__':
    # Start the Flask application on port 5002
    app.run(port=5002)
