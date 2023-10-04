import tensorflow as tf
import cv2 
import numpy as np
import matplotlib.pyplot as plt



def predictions(frame, loaded_model):

    faces = []
    imaget_path = "./prueba/neutral.png"
    class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']
    #face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(frame, (48, 48))

    face2 = tf.keras.preprocessing.image.img_to_array(face)
    face2 = np.expand_dims(face2,axis=0)

    faces.append(face2)


    # Realizar inferencia con el modelo cargado
    predictions = loaded_model.predict(faces) 
    

    return class_names[np.argmax(predictions)]
