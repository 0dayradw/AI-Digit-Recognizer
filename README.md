# ✏️ Handwritten Digit Recognition with CNN

An interactive Pygame app that recognizes digits drawn by hand using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

---

## 📁 Project Structure

```
digit_rec/
├── Lib/
│   ├── predict_game2.py         # Main game interface
│   └── constants.py             # UI constants (colors, sizes, etc.)
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
git clone https://github.com/0dayradw/AI-Digit-Recognizer
cd AI-Digit-Recognizer
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
| ![Canvas]() | ![Model]() |

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
