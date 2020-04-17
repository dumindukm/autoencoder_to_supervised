
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
import cv2
# https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726
# %matplotlibe inline


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')


def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")
    print('ssss')


(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_x = x_train.reshape(60000, 784) / 255
val_x = x_test.reshape(10000, 784) / 255

autoencoder = Sequential()
autoencoder.add(Dense(512,  activation='elu', input_shape=(784,)))
autoencoder.add(Dense(128,  activation='elu'))
autoencoder.add(Dense(10,    activation='linear', name="bottleneck"))
autoencoder.add(Dense(128,  activation='elu'))
autoencoder.add(Dense(512,  activation='elu'))
autoencoder.add(Dense(784))  # ,  activation='sigmoid'
autoencoder.compile(loss='mean_squared_error', optimizer=Adam())
# trained_model = autoencoder.fit(
#     train_x, train_x, batch_size=1024, epochs=10, verbose=1, validation_data=(val_x, val_x))

# autoencoder.save('checkpoints/encoder_model.h5')
autoencoder = load_model('checkpoints/encoder_model.h5')

# encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
# encoded_data = encoder.predict(train_x)  # bottleneck representation
# decoded_output = autoencoder.predict(train_x)        # reconstruction
# encoding_dim = 10

# # return the decoder
# encoded_input = Input(shape=(encoding_dim,))
# decoder = autoencoder.layers[-3](encoded_input)
# decoder = autoencoder.layers[-2](decoder)
# decoder = autoencoder.layers[-1](decoder)
# decoder = Model(encoded_input, decoder)

# testing

# print(val_x.shape)

# plot_image(val_x[0])

cv2.imshow('ori', val_x[0].reshape(28, 28))

decoded_output = autoencoder.predict([[val_x[0]]])

# plot_image(decoded_output.reshape(28, 28))
# decoded_output *= 255
cv2.imshow('pred', decoded_output.reshape(28, 28))
print(val_x[0])
print(decoded_output)
print('done')

cv2.waitKey(0)
cv2.destroyAllWindows()
