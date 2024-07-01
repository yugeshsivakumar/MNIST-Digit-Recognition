from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
model = load_model('mnist_model.h5')  # Update this path to your saved model

@app.route('/')
def index():
    return render_template('index.html')  # Serve the index.html file

@app.route('/predict', methods=['POST'])
def predict_digit():
    drawing_data = request.json['drawing']  # Expect JSON input with 'drawing' key
    
    processed_data = preprocess_input(drawing_data)
    
    prediction = model.predict(processed_data)
    predicted_digit = np.argmax(prediction)
    
    return jsonify({'predicted_digit': int(predicted_digit)})

def preprocess_input(drawing_data):
    drawing_data = np.array(drawing_data, dtype=np.uint8)  # Convert list to NumPy array
    drawing_data = drawing_data.reshape(200, 200)  # Ensure it is reshaped to 200x200
    img = cv2.resize(drawing_data, (28, 28))  # Resize to 28x28
    img = img.astype('float32') / 255.0  # Normalize the image to [0,1] range
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

if __name__ == '__main__':
    app.run(debug=True)
