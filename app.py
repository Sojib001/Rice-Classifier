# app.py

from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from io import BytesIO

# Load the trained model
model_path = 'rice_model.h5'  # Make sure your model is saved in a .h5 format (Keras model)
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file uploaded")
    
    file = request.files['file']
    
    # If no file is selected or the file has no allowed extension
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', prediction_text="Invalid file type. Please upload a valid image.")
    
    # Convert the file to a BytesIO object
    img = image.load_img(BytesIO(file.read()), target_size=(256, 256))  # Match your model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Normalize if the model was trained with normalized data
    img_array /= 255.0
    
    # Predict the class
    predictions = model.predict(img_array)
    print(predictions)
    # Assuming your model has 2 classes or you have a multi-class model
    predicted_class = np.argmax(predictions, axis=1)  # For multi-class classification
    class_names = ['A', 'B', 'C', 'D', 'E']  # Modify based on your model's class names
    predicted_label = class_names[predicted_class[0]]
    
    return render_template('index.html', prediction_text=f'Prediction: {predicted_label}')
if __name__ == "__main__":
    app.run(debug=True)