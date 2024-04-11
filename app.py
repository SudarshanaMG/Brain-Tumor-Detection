from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import secrets
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



app = Flask(__name__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


model1 = load_model('models/cnn224.h5')
model2 = load_model('models/vgg16.h5')
model3 = load_model('models/resnet_model.h5')
model4 = load_model('models/inceptionv3_model.h5')



def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))
    img = img.reshape((1,224,224,3))

    # img = np.expand_dims(img, axis=0)
    # img = img.astype('float32')/255
   
    preds = model.predict(img)
    if model==model1:
        if preds == 0:
            return "No tumor"
        else:
            return "Tumor"
        
    pred = np.argmax(preds,axis = 1)
    if pred == 0:
        return "No tumor"
    else:
        return "Tumor"

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']


        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        predictions = {
            'model1': model_predict(file_path, model1),
            'model2': model_predict(file_path, model2),
            'model3': model_predict(file_path, model3),
            'model4': model_predict(file_path, model4)
        }
        os.remove(file_path)
        return jsonify(predictions)
    return None



if __name__ == '__main__':
    app.run(debug=True)
