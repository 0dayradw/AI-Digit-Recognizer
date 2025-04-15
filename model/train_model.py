from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.datasets import mnist
import tensorflow as tf

from Lib.constants import *

def train_model():
    # Încărcăm setul de date MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Redimensionăm imaginile la 64x64 și normalizăm valorile pixelilor
    x_train = tf.image.resize(x_train[..., tf.newaxis], [64, 64]) / 255.0
    x_test = tf.image.resize(x_test[..., tf.newaxis], [64, 64]) / 255.0

    # Construim modelul
    model = Sequential()

    # Blocul 1                           # Convoluție cu 32 de filtre
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(BatchNormalization())             # Normalizare pe loturi
    model.add(MaxPooling2D(pool_size=(2, 2)))   # Reducere dimensională (pooling)
    model.add(Dropout(0.3))                     # Eliminare aleatorie a neuronilor (regularizare)

    # Blocul 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Blocul 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Blocul 4
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Trecem la straturile dense (fully connected)
    model.add(GlobalAveragePooling2D())

    # Straturi Dense
    model.add(Dense(128, activation='relu'))  # Strat dens cu 128 de neuroni
    model.add(Dropout(0.5))  # Regularizare



    # Strat de ieșire
    model.add(Dense(10, activation='softmax'))  # 10 clase pentru cifre (0-9)

    # Compilarea modelului
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Afișează un rezumat al modelului
    model.summary()

    # Antrenarea modelului
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Evaluarea performanței pe setul de testare
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nAcuratețea pe setul de test:', test_acc)

    # Salvăm modelul antrenat
    model.save("digit_model.keras")


if __name__ == '__main__':
    train_model()

