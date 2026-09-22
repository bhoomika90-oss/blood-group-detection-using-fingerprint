"# blood-group-detection-using-fingerprint" 
# Blood Group Detection Using Fingerprint

## Overview

Blood Group Detection Using Fingerprint is an experimental machine learning project that uses a Convolutional Neural Network (CNN) to classify fingerprint images into eight blood-group labels.

The system provides two input methods:

- Camera capture
- Fingerprint image upload

The captured or uploaded fingerprint image is preprocessed and passed to the trained CNN model.

## Blood Group Classes

The model classifies the following labels:

- A-
- A+
- AB-
- AB+
- B-
- B+
- O-
- O+

## Technologies Used

- Python
- Flask
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Pillow
- Scikit-learn
- HTML
- CSS
- JavaScript

## Machine Learning Model

The project uses a Convolutional Neural Network (CNN).

Input image:

- Grayscale
- 128 × 128 pixels
- Normalized pixel values

CNN architecture includes:

- Convolutional layers
- Max Pooling layers
- Fully Connected layer
- Dropout
- Softmax output layer

## Dataset

The dataset contains approximately 6000 fingerprint images distributed across eight blood-group classes.

The dataset is intentionally excluded from this GitHub repository because of its size.

## Features

### Camera Capture

Users can use their camera to capture a fingerprint image directly from the web application.

### File Upload

Users can select an existing fingerprint image from their computer.

### CNN Prediction

The trained CNN predicts one of the eight dataset labels and displays class probabilities.

## Project Structure

```text
Blood Group Detection Using Fingerprint/
│
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── preprocess.py
├── split_dataset.py
├── verify_processed.py
├── check_dataset.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
│
├── model/
│   └── blood_group_cnn.h5
│
├── notebooks/
│
├── requirements.txt
├── .gitignore
└── README.md"# blood-group-detection-using-fingerprint" 
