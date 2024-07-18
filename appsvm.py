import os
from flask import Flask, render_template, request
from PIL import Image
from joblib import load
from skimage.feature import hog
import cv2
import torch
import numpy as np
import pandas as pd

app = Flask(__name__)

disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

loaded_svm_model = load("SVM_PLANT_DISEASE.joblib")


def predict_disease(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_test = hog(image, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')

    y_test_pred = loaded_svm_model.predict(x_test.reshape(1, -1))

    diseases = {
        0: "Apple-Black-rot",
        1: "Blueberry-healthy",
        2: "Cherry-Powdery-mildew",
        3: "Corn-(maize)-Common-rust",
        4: "Grape-Black-rot",
        5: "Orange-Haunglongbing-(Citrus_greening)",
        6: "Peach-Bacterial-spot",
        7: "Pepper-bell-Bacterial-spot",
        8: "Potato-Early-blight",
        9: "Raspberry-healthy",
        10: "Soybean-healthy",
        11: "Squash-Powdery-mildew",
        12: "Strawberry-Leaf-scorch",
        13: "Tomato-Early-blight",
        14: "Tomato-Tomato-Yellow-Leaf-Curl-Virus",
    }

    predicted_disease = diseases[y_test_pred[0]]

    return predicted_disease

ind_find = {
        "Apple-Black-rot":1,
        "Blueberry-healthy":5,
        "Cherry-Powdery-mildew":6,
        "Corn-(maize)-Common-rust":9,
        "Grape-Black-rot":12,
        "Orange-Haunglongbing-(Citrus_greening)":16,
        "Peach-Bacterial-spot":17,
        "Pepper-bell-Bacterial-spot":19,
        "Potato-Early-blight":21,
        "Raspberry-healthy":24,
        "Soybean-healthy":25,
        "Squash-Powdery-mildew":26,
        "Strawberry-Leaf-scorch":27,
        "Tomato-Early-blight":30,
        "Tomato-Tomato-Yellow-Leaf-Curl-Virus":36
        
    }

@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = predict_disease(file_path)
        pred = ind_find[pred]
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)


@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))


if __name__ == '__main__':
    app.run(debug=True)
