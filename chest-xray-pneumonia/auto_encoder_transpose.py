# https: // www.machinecurve.com/index.php/2019/12/11/upsampling2d-how-to-use-upsampling-with-keras/
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Conv2DTranspose
# import matplotlib.pyplot as plt
import numpy as np

# Model configuration
img_width, img_height = 28, 28
batch_size = 25
no_epochs = 25
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Reshape data
input_train = input_train.reshape(
    input_train.shape[0], img_width, img_height, 1)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Create the model
# model = Sequential()
# model.add(Conv2D(8, (2, 2), activation='relu',
#                  kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
# model.add(MaxPooling2D((2, 2), padding='same'))
# model.add(Conv2D(8, (2, 2), activation='relu',
#                  kernel_initializer='he_uniform', padding='same'))
# model.add(MaxPooling2D((2, 2), padding='same'))
# model.add(Conv2D(8, (2, 2), strides=(2, 2), activation='relu',
#                  kernel_initializer='he_uniform', padding='same'))
# model.add(Conv2D(8, (2, 2), activation='relu',
#                  kernel_initializer='he_uniform', padding='same'))
# model.add(UpSampling2D((2, 2), interpolation='bilinear'))
# model.add(Conv2D(8, (2, 2), activation='relu'))
# model.add(UpSampling2D((2, 2), interpolation='bilinear'))
# model.add(Conv2D(8, (2, 2), activation='relu',
#                  kernel_initializer='he_uniform', padding='same'))
# model.add(UpSampling2D((2, 2), interpolation='bilinear'))
# model.add(Conv2D(1, (2, 2), activation='sigmoid', padding='same'))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_normal', input_shape=input_shape))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_normal'))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_normal'))
model.add(Conv2DTranspose(8, kernel_size=(3, 3),
                          activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(16, kernel_size=(3, 3),
                          activation='relu', kernel_initializer='he_normal'))
model.add(Conv2DTranspose(32, kernel_size=(3, 3),
                          activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))

model.summary()

# Compile and fit data
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(input_train, input_train,
          epochs=no_epochs,
          batch_size=batch_size,
          validation_split=validation_split)
