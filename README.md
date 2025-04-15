Handwritten Digit Recognition
This project is a Handwritten Digit Recognition model built using TensorFlow and trained on the MNIST dataset. The model recognizes handwritten digits from images drawn using a Pygame-based graphical interface. Once a digit is drawn, the model will predict the digit and display the confidence level of the prediction.

Model Architecture
The model architecture consists of multiple Convolutional Neural Networks (CNNs) layers for feature extraction and dense layers for classification. Below is the architecture for the model:

Model Details
  Convolutional Layers: Extract features from the image using filters.
  Pooling Layers: Reduce the dimensionality of the data and prevent overfitting.
  Dense Layers: Classify the extracted features into one of the 10 digit classes (0-9).
  Dropout Layers: Used for regularization to avoid overfitting.

![image](https://github.com/user-attachments/assets/e9a6f993-3946-4533-bcb1-de425e3ae733)

How the Application Works
1. Drawing the Digit:
  Users can draw a digit on a canvas provided by the Pygame-based interface.
  The digit is drawn in a 64x64 pixel canvas, which is then processed and resized to match the model's input dimensions.
2. Prediction:
  After drawing, the user can click the "Predict" button, which triggers the model to predict the drawn digit.
  The prediction and the confidence level of the prediction will be displayed on the screen.

Example Predictions
  Here are some example predictions from the model:
    ![Screenshot (482)](https://github.com/user-attachments/assets/8774cb35-62c3-467d-92c5-c79ef943809b)

Requirements

  You can the install them via pip using the requirements.txt file:
    pip install -r requirements.txt

