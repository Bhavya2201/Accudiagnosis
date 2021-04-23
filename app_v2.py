import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
#run_with_ngrok(app)

PATH_MODEL = 'models/pathmnist.h5'
DERMA_MODEL = 'models/dermamnist.h5'
OCT_MODEL = 'models/octmnist.h5'
PNEUMONIA_MODEL = 'models/pneumoniamnist.h5'
BREAST_MODEL = 'models/breast.h5'


def model_select():
    val=request.form.get("a")
    val=int(val)
    #path_list=['pneumoniamnist.h5', 'breast.h5', 'dermamnist.h5', 'pathmnist.h5', 'octmnist.h5']
    path_list=[PNEUMONIA_MODEL, BREAST_MODEL, DERMA_MODEL, PATH_MODEL, OCT_MODEL]
    if(val == 0 or val == 1 or val == 4):
        reshape_val = 1
    else:
        reshape_val = 3
    #print(path_list[val])
    #print(type(path_list[val]))
    model = load_model(path_list[val])
    #print(path_list[val])
    model._make_predict_function()
    #print(reshape_val)
    return model,reshape_val

def class_select():
    pneumonia_class = ["normal",  "pneumonia"]
    breast_class = ["malignant", "normal, benign"]
    path_class = [ "adipose",  "background",  "debris",  "lymphocytes",  "mucus", "smooth muscle",  "normal colon mucosa",  "cancer-associated stroma",  "colorectal adenocarcinoma epithelium"]
    oct_class = ["choroidal neovascularization",  "diabetic macular edema",  "drusen",  "normal"]
    derma_class = ["actinic keratoses and intraepithelial carcinoma", "basal cell carcinoma", "benign keratosis-like lesions",  "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions"]
    val = request.form.get("a")
    val=int(val)
    class_labels=None
    if(val == 0):
        class_labels=pneumonia_class
    elif(val == 1):
        class_labels=breast_class
    elif (val == 2):
        class_labels = derma_class
    elif (val == 3):
        class_labels = path_class
    elif (val == 4):
        class_labels = oct_class

    return class_labels

def model_predict(img_path, model, reshape_val):
    val = request.form.get("a")
    val = int(val)
    if(val==2):
        img = image.load_img(img_path, target_size=(32, 32))
        x = image.img_to_array(img)
        final = x.reshape(1, 32, 32, reshape_val)
    elif(val == 0 or val == 1 or val == 4):
        img = image.load_img(img_path, target_size=(28, 28),grayscale=True)
        x = image.img_to_array(img)
        #print(x.shape)
        final = x.reshape(1, 28, 28, reshape_val)
    else:
        img = image.load_img(img_path, target_size=(28, 28))
        x = image.img_to_array(img)
        #print(x.shape)
        final = x.reshape(1, 28, 28, reshape_val)
    preds = np.argmax(model.predict(final))
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
             basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        model, reshape_val=model_select()
        #print(reshape_val)
        preds = model_predict(file_path, model, reshape_val)
        class_labels = class_select()
        result = class_labels[preds]
        #return result
        return render_template('index.html',
                               prediction_text='Diagnosis is {} '.format(result))
    return None

if __name__ == '__main__':
    app.run(debug=True)

