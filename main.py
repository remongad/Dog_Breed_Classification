from sklearn.datasets import load_files       
import numpy as np
from glob import glob
import random
import cv2                
import matplotlib.pyplot as plt  
from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import ImageFile  
from tensorflow.keras.applications.resnet50 import ResNet50
from extract_bottleneck_features import *
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
import pickle

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def get_resnet_50_model(img_path):
    bottleneck_features = extract_Resnet50(path_to_tensor(img_path))
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=bottleneck_features.shape[1:]))
    Resnet50_model.add(Dense(133, activation='softmax'))
    Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
    return Resnet50_model, bottleneck_features

def Resnet50_preed_predict(img_path):
    with open('dog_names.pkl', 'rb') as f:
        dog_names = pickle.load(f)
    Resnet50_model, bottleneck_features = get_resnet_50_model(img_path)
    
    predicted_labels = Resnet50_model.predict(bottleneck_features)
    
    breed = dog_names[np.argmax(predicted_labels)]
    
    return breed.split('.')[1]

def predict_preed(img_path):
    
    is_human = face_detector(img_path)
    is_dog = dog_detector(img_path)
    
    if is_dog:
        print('Dog Detected')
        print('The dog preed is: {}'.format(Resnet50_preed_predict(img_path)))
    elif is_human:
        print('Human Detected')
        print('The Resempling Human preed is: {}'.format(Resnet50_preed_predict(img_path)))
    else:
        print('Neither Human Nor Dog')

predict_preed('C:\Personal\dog_preed_classification\Dog_Breed_Classification\images\human1.jpg')