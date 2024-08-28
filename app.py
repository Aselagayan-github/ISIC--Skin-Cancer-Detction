from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = load_model(r"C:\Users\DELL\Desktop\ISIC Skin Cancer 2024\model\resnet_model.hdf5")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/showresult', methods=['POST'])
def show_result():
    if 'pic' not in request.files:
        return redirect(url_for('home'))

    file = request.files['pic']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        img = Image.open(file).resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        prediction = model.predict(img_array)
        result = "Malignant" if prediction[0][0] > 0.5 else "Benign"

        # You can add more info based on the result here
        info = "This lesion is classified as {}.".format(result)

        return render_template('result.html', result=result, info=info)

if __name__ == '__main__':
    app.run(debug=True)
