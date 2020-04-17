from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
# from keras.datasets import mnist
import numpy as np
import glob
import cv2
import pandas as pd
import custom_data_generator_for_encoder as cdatagen

pd.set_option('display.max_rows', 5000)

image_shape_channel = (28, 28, 3)
image_shape = (image_shape_channel[0], image_shape_channel[1])


def generator(samples, batch_size=32, shuffle_data=True, resize=224, input_shape=(28, 28, 3)):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        # shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img = cv2.imread(batch_sample)
                img = cv2.resize(img, (input_shape[0], input_shape[1]))
                img = img/255

                # apply any kind of preprocessing                img = cv2.resize(img,(resize,resize))
                # Add example to arrays
                X_train.append(img)
                # y_train.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            # X_train = np.array(X_train)
            # y_train = np.array(y_train)
            X_train = np.reshape(
                X_train, (-1, input_shape[0], input_shape[1], input_shape[2]))
            # The generator-y part: yield the next training batch
            yield X_train, X_train


def custom_train_image_generator(img_path, input_shape, batch_size, caller):

    img_path_list = glob.glob(img_path)[0:6]
    img_count = 0
    index = 0
    indexes = np.arange(len(img_path_list))
    # print(len(img_path_list))
    while True:
        images = []

        # Generate indexes of the batch
        indexes = indexes[index*batch_size:(index+1)*batch_size]

        # Find list of IDs
        image_paths_temp = [img_path_list[k] for k in indexes]
        for i, img_path in enumerate(image_paths_temp):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (input_shape[0], input_shape[1]))
            img = img/255
            # print(img.shape)
            # images.extend(np.reshape(img, (28, 28, 3)))
            images.append(img)

        reshaped_images = np.reshape(
            images, (-1, input_shape[0], input_shape[1], input_shape[2]))
        index += 1
        yield (reshaped_images, reshaped_images)

        # for path in range(batch_size):

        #     img = cv2.imread(img_path_list[img_count])
        #     img = cv2.resize(img, (input_shape[0], input_shape[1]))
        #     img = img/255
        #     # print(img.shape)
        #     # images.extend(np.reshape(img, (28, 28, 3)))
        #     images.append(img)
        #     # images.append(img)
        #     img_count += 1

        #     # if img_count % batch_size == 0 or img_count >= len(img_path_list):
        #     #     break

        #     print(np.shape(images), caller)
        #     reshaped_images = np.reshape(
        #         images, (-1, input_shape[0], input_shape[1], input_shape[2]))
        #     yield (reshaped_images, reshaped_images)


# (x_train, _), (x_test, _) = mnist.load_data()


# adapt this if using `channels_first` image data format
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# adapt this if using `channels_first` image data format

input_img = Input(shape=image_shape_channel)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print(autoencoder.summary())

# autoencoder.fit(x_train, x_train,
#                 epochs=10,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_test, x_test),
#                 )  # callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]

train_img_dir = 'chest_xray/train/*/*.jpeg'
test_img_dir = 'chest_xray/test/*/*.jpeg'
val_img_dir = 'chest_xray/val/*/*.jpeg'
batch_size = 1

# print("AAAAA", len(glob.glob(train_img_dir)[0:6]))

# a = next(custom_train_image_generator(train_img_dir,
#                                       batch_size=batch_size, input_shape=image_shape_channel, caller='train'))

# print(a)
# generator
# #img_path, input_shape, batch_size, caller
# autoencoder.fit_generator(custom_train_image_generator(train_img_dir, batch_size=batch_size, input_shape=image_shape_channel, caller='train'),
#                           steps_per_epoch=2,
#                           epochs=1,
#                           # callbacks=callbacks,
#                           validation_data=custom_train_image_generator(
#     val_img_dir, batch_size=batch_size, input_shape=image_shape_channel, caller='val'),
#     validation_steps=3
# )


X_img_train = []
paths = glob.glob(train_img_dir)[0:2]
for im in paths:
    img = cv2.imread(im)
    img = cv2.resize(img, (image_shape_channel[0], image_shape_channel[1]))
    img = img/255
    # print(img.shape)
    # images.extend(np.reshape(img, (28, 28, 3)))
    X_img_train.append(img)

X_img_train = np.reshape(
    X_img_train, (-1, image_shape_channel[0], image_shape_channel[1], image_shape_channel[2]))

X_val_img_train = []
paths = glob.glob(val_img_dir)[0:6]
for im in paths:
    img = cv2.imread(im)
    img = cv2.resize(img, (image_shape_channel[0], image_shape_channel[1]))
    img = img/255
    # print(img.shape)
    # images.extend(np.reshape(img, (28, 28, 3)))
    X_val_img_train.append(img)

X_val_img_train = np.reshape(
    X_val_img_train, (-1, image_shape_channel[0], image_shape_channel[1], image_shape_channel[2]))

autoencoder.fit(X_img_train, X_img_train,
                # batch_size=2,
                # steps_per_epoch=2,
                epochs=2,
                # callbacks=callbacks,
                validation_data=(X_val_img_train, X_val_img_train),
                # validation_steps=3
                )


# autoencoder.fit_generator(generator(glob.glob(train_img_dir), batch_size=batch_size, input_shape=image_shape_channel),
#                           steps_per_epoch=2,
#                           epochs=1,
#                           # callbacks=callbacks,
#                           validation_data=generator(
#     glob.glob(val_img_dir), batch_size=batch_size, input_shape=image_shape_channel),
#     validation_steps=3
# )


# history = autoencoder.fit_generator(cdatagen.CustomImageDataGeneratorForEncode(train_img_dir, batch_size=batch_size, dim=image_shape_channel),
#                                     steps_per_epoch=3,
#                                     epochs=1,
#                                     # callbacks=callbacks,
#                                     validation_data=cdatagen.CustomImageDataGeneratorForEncode(
#                                         val_img_dir, dim=image_shape_channel, batch_size=batch_size),
#                                     validation_steps=1
#                                     )

autoencoder.save('autoencoder.h5')
autoencoder = load_model('autoencoder.h5')
# decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


print('completed.')
