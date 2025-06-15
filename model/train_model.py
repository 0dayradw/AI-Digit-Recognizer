from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.datasets import mnist
import tensorflow as tf

from Lib.constants import *

def train_model():
    # Loading the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Resize each image at 64x64
    x_train = tf.image.resize(x_train[..., tf.newaxis], [64, 64]) / 255.0
    x_test = tf.image.resize(x_test[..., tf.newaxis], [64, 64]) / 255.0

    # Model arhitecture
    model = Sequential()

    # First Block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(BatchNormalization())             
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Dropout(0.3))                    

    # Second Block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Thrid Block
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Fourth Block
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Dense (fully connected)  
    model.add(GlobalAveragePooling2D())

    # Dense layers
    model.add(Dense(128, activation='relu'))  
    model.add(Dropout(0.5)) 



    # Output layer
    model.add(Dense(10, activation='softmax'))  # 10 class for digits (0-9)

    #  Model compilation
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Model summary
    model.summary()

    # Training the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Check the accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nAcuratețea pe setul de test:', test_acc)

    # Saving the model 
    model.save("digit_model.keras")


if __name__ == '__main__':
    train_model()

