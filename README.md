# Dog Breed Detector Using Flask web application

## Introduction
This is the capstone project for the data scientist nanodegree. this project is using flask to build a web app that you can use to upload an image and it identify the breed of the dog if a dog exists in this image. if there is a human in the image then it detects it and show the dog breed that resemble that resemble this human face

## Repository Structure
you can find the flask code in the app.py file which call all necessary files to run the web app and instantiate a local server connection so that you can use it to upload an image and experiement with the web app as much as you want by simply uploading an image from your local computer. You can also have a look at the process by which this deep learning model was created in the python notebook dog_app.ipynb 

## How it works
* Clone this repository git lfs clone https://github.com/Andrew-Maged/DSND-Dog-Breed-Classifier
* Open project's directory cd Dog_Breed_Classification
* Install all necessary dependencies pip install -r requirements.txt
* Run application python app.py
* Open http://127.0.0.1:5000/ on your browser
* Click the file select button and select test image for classifier.

## Prerequisite
* Python 3.7+
* Keras
* OpenCV
* Matplotlib
* NumPy
* glob
* tqdm
* Scikit-Learn
* Flask
* Tensorflow
