from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import cv2
import numpy as np


app = Flask(__name__)
#loading the trained model
model = tf.keras.models.load_model(r"C:\lux academy datacamp\Machine Learning\detection_model\animal_detection_model1.h5")

#the preprocessing function
def preprocess(image):
    image = cv2.resize(image, (150,150))
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    return image

@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')

#defining predictions function
@app.route('/predict', methods=['POST'])
def predict():
    #GETTING THE IMAGE FROM REQUEST
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(),
                                       np.uint8),
                                       cv2.IMREAD_COLOR)
    #PREPROCESS IMAGE
    image = preprocess(image)
    #Make prediction
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = classnames[class_index]

    #return a json object
    return jsonify({'class_name': class_name})

if __name__ == '__main__':
    app.run(debug=True)