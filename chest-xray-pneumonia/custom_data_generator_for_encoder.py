# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import tensorflow.keras
import cv2
import glob


class CustomImageDataGeneratorForEncode(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, image_paths,  batch_size=32, dim=(32, 32, 32), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.image_paths = glob.glob(image_paths)[0:6]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 2  # int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        image_paths_temp = [self.image_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(image_paths_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        images = []
        # Generate data
        for i, img_path in enumerate(image_paths_temp):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.dim[0], self.dim[1]))
            img = img/255
            # print(img.shape)
            # images.extend(np.reshape(img, (28, 28, 3)))
            images.append(img)

        reshaped_images = np.reshape(
            images, (-1, self.dim[0], self.dim[1], self.dim[2]))
        return reshaped_images, reshaped_images
