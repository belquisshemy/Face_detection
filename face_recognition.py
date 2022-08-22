# Import all the necessary files!

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.python.keras.backend import set_session
import cv2
import json
import numpy as np
from datetime import datetime
import glob
train_path='E:/face_recgnotion/venv/train'
test_path='E:/face_recgnotion/venv/test'
model = tf.keras.models.load_model('facenet_keras.h5')


def encode(path, model):
    img = cv2.imread(path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
       face = img[y:y + h, x:x + w]
       dim = (160, 160)
       img = cv2.resize(face, dim, interpolation = cv2.INTER_AREA)
       x_train = np.array([img])
       embedding = model.predict(x_train)
    return embedding

database = {}
for filename in glob.glob('E:/face_recgnotion/venv/train/*.jpg'):
    filename = os.path.basename(filename)
    name=os.path.splitext(filename)[0]
    database[name] = encode(train_path +'/'+filename, model)


def verify(image_path, database, model):
    result_encoded = encode(image_path, model)
    min_dist = 1000
    #Looping over the names and encodings in the database.
    for (name, dist_database) in database.items():
        dist = np.linalg.norm(result_encoded - dist_database)
        print(dist)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 10:
        print("Not in the database.")
    else:
        print ("It is " + str(identity) + ", and the distance is " +
    str(min_dist))
    return min_dist, identity


distance, person_name = verify(test_path + '/9.jpg', database , model)





