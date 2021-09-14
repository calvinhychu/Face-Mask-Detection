# Face Mask Detection
Face Mask Detection is an application using deep neural network and computer vision to detect if a person is wearing a face mask.

## Demo
<img src="./misc/maskdemo.gif"/>

## How it works?
This application uses tensorflow and keras to build a model to recognize a person wearing face mask with deep neural network. Dataset of images used to build the model can be found [here] (https://www.kaggle.com/sumansid/facemask-dataset). More images are used to train the model via data augmentation to avoid overfitting. The model uses Convolutional Neural Network (CNN) as the class of neural network for image classification. VGG16 is used the base model and is tinkered with more neural network layer to achieve minimal loss and higher accuracy rate on both validation set and test set. 

OpenCV is used to build the facial recognition part of the application. It captures frontal face image/frame in real time for the trained model to predict if the user is wearing a mask or not. If for 90 frames (around 5 seconds) in a row where the model predicts the user is wearing a mask with more than 99% certainty, the application will deem that the user is wearing a mask.

## Usage
1. Install python libraries in requirements.txt
2. Use the trained model from this repository or trained a new model by running <b><i>model_training.py</i></b>
3. Replace the directory for the classifier in <b><i>facemask_detection.py</i></b>
3. Run <b><i>facemask_detection.py</i></b> 

## Model Training 
The model uses Convolutional Neural Network (CNN) as the class of neural network for image classification. A sequential layered model is used with VGG16 as the initial/base model and is tinkered with more neural network layer to achieve minimal loss and higher accuracy rate on both validation set and test set. An early stoppage monitor is used to ensure the model will stop training if there is no improvement in validation loss for 2 straight epochs.

Loss function used is Binary Crossentropy as the true labels and predicted labels are either "Mask" or "No Mask".

After trial with Adam and Stochastic gradient descent (SGD) as optimizer to change weight and bias, Adam is chosen as the optimizer with better results.

Training and testing results are shown in plots below.

<img src="./misc/training_result.png"/>

## Computer Vision
Built-in frontal face classifier in OpenCV is used to detect human face via a webcam. Every facial detected frame will be captured and resized and converted to a numpy array to fit the trained model. The model will then predict if the human in the captured frame has a mask on or not with a confidence percentage. If for 90 frames in a row where the model predicted the user has a mask on with more than 99% confidence, a message will be on screen to signify the user is wearing a mask and wearing it properly.


## Disclaimer
Face Mask Detection is only intended for educational purposes and all dataset images are property of Kaggle.