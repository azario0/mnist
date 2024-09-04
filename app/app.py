from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('mnist_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) 
    
    # Crop to the bounding box of the non-zero pixels
    non_empty_columns = np.where(image_array.min(axis=0) < 255)[0]
    non_empty_rows = np.where(image_array.min(axis=1) < 255)[0]
    crop_box = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    
    # Crop the image to the bounding box
    image_array = image_array[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1]
    
    # Resize the cropped image back to 28x28
    image = Image.fromarray(image_array).resize((28, 28), Image.LANCZOS)
    
    # Normalize pixel values
    image_array = np.array(image).reshape(1, -1) / 255.0


    # Make prediction
    prediction = model.predict(image_array)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)