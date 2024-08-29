from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
model_path = './model/resnet50_model.h5'
model = load_model(model_path)

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    image = Image.open(io.BytesIO(image))
    image = image.resize((224, 224))  # Resize to the input size of your model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/showresult', methods=['POST'])
def showresult():
    if 'pic' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['pic']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        image = file.read()
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        # Assuming the model outputs a probability and you have a threshold for classification
        result = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'
        info = 'Please consult a dermatologist for further diagnosis.' if result == 'Malignant' else 'The result appears benign, but always seek professional medical advice.'

        return render_template('result.html', result=result, info=info)

if __name__ == '__main__':
    app.run(debug=True)
