# Face-Recognition-Using-Open-CV-CNN
This project implements face recognition using OpenCV and CNNs. It involves detecting faces in images or video streams with OpenCV, training a Convolutional Neural Network (CNN) for accurate face recognition, and achieving real-time performance for authentication or surveillance purposes.
 

## Project Overview  
This project implements a face recognition system using OpenCV and a Convolutional Neural Network (CNN). It captures face data, preprocesses and trains a CNN model for face recognition, and performs real-time recognition with a webcam. The system is ideal for authentication and surveillance applications.  

---

## Features  
1. Face Data Collection:
   - Captures face data using OpenCV's Haar Cascade Classifier.  
   - Saves images in a structured dataset for training.  
2. Data Preprocessing: 
   - Normalizes images, converts them to grayscale, and resizes them to a consistent size (100x100).  
3. CNN Model Training: 
   - Builds a CNN model with TensorFlow/Keras to classify faces.  
   - Trains the model with captured face data and validates its performance.  
4. Real-Time Recognition:
   - Uses the trained model to identify faces with bounding boxes and confidence levels.  

---

## Tools & Libraries Used  
Python  
OpenCV for face detection and real-time video processing.  
TensorFlow/Keras for building and training the CNN model.  
NumPy for data manipulation.  
scikit-learn for data splitting.  
##Over view
1. Face Data Collection
Captures face images using Haar Cascade Classifier.
Saves cropped face images in a folder named after the user.
2. Data Preprocessing
Converts images to grayscale and resizes them to 100x100 pixels.
Normalizes image data to improve CNN performance.
3. CNN Model Training
Builds a CNN with layers for convolution, pooling, flattening, and dense connections.
Uses softmax activation for multi-class classification.
4. Real-Time Recognition
Detects faces in webcam frames, preprocesses them, and predicts the user's name and confidence score.
Outputs
Face Data Collection: Stores 30 images per user in the dataset folder.
Training: Displays training accuracy and loss.
Real-Time Recognition: Displays recognized names and confidence scores with bounding boxes.

  
