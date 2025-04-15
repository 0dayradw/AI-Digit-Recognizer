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
    ![Screenshot (483).png](..%2Fimages%2FScreenshot%20%28483%29.png)

Requirements

  You can the install them via pip using the requirements.txt file:
    pip install -r requirements.txt

# ✏️ Handwritten Digit Recognition with CNN

An interactive Pygame app that recognizes digits drawn by hand using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

---

## 📁 Project Structure

```
digit_rec/
├── Lib/
│   ├── predict_game2.py         # Main game interface
│   ├── constants.py             # UI constants (colors, sizes, etc.)
│   └── utils.py (optional)      # Utility functions
├── model/
│   └── digit_model.keras        # Trained CNN model
├── images/
│   ├── architecture.png         # Model architecture
│   └── sample_prediction.png    # Example prediction
├── train_model.py               # Training script
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/digit_rec.git
cd digit_rec
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Train the Model (if not already trained)

```bash
python train_model.py
```

Or use the existing pre-trained model in `model/digit_model.keras`.

---

## 🎮 Run the App

```bash
python main.py
```

Use your mouse to draw digits, press **Predict**, and the model will recognize the number.

---

## 📷 Screenshots

| Drawing Canvas        | Prediction Output         |
|-----------------------|---------------------------|
| ![Canvas](images/sample_prediction.png) | ![Model](images/architecture.png) |

---

## 🧠 Model Details

- **Framework**: TensorFlow / Keras
- **Dataset**: MNIST (60,000 training, 10,000 testing images)
- **Architecture**: CNN (Convolutional Neural Network)
- **Input Size**: 64x64 grayscale images

---

## 💻 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Pygame
- NumPy
- Matplotlib

---

## 📄 License

MIT License – Feel free to use, modify, and share!

---

## 🙌 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- TensorFlow/Keras tutorials
