#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import tensorflow as tf
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#import imageio as im
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.preprocessing import image as imagepre
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import cv2
# In[]

pd.set_option('display.max_rows', 5000)

image_shape_channel = (64, 64, 3)
image_shape = (image_shape_channel[0], image_shape_channel[1])

# In[12]:


# In[13]:

input = Input(shape=image_shape_channel)
# First Layer
x = Conv2D(128, (3, 3), padding='same',
           input_shape=image_shape_channel, activation='relu')(input)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


# Step 3 - Flattening
x = Flatten()(x)

# Step 4 - Full connection
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(units=2, activation='softmax')(x)


# In[14]:


classifier = Model(input, output)

classifier.summary()


classifier.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


# In[16]:
train_img_dir = 'chest_xray/train'
test_img_dir = 'chest_xray/test'
val_img_dir = 'chest_xray/val'

batch_size = 10

train_datagen = ImageDataGenerator(rescale=1./255,
                                   # validation_split=0.2,
                                   # height_shift_range=0.5
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_img_dir,
                                                 target_size=image_shape,
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 )  # subset='training'

val_set = test_datagen.flow_from_directory(val_img_dir,
                                           target_size=image_shape,
                                           batch_size=2,
                                           class_mode='categorical',
                                           )  # subset='validation'


checkpointer = ModelCheckpoint(filepath="best_weights_cnn.hdf5",
                               monitor='val_acc',
                               verbose=1,
                               save_best_only=True)

callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    # Write TensorBoard logs to `./logs` directory
    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    #tf.keras.callbacks.ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='min')
]


# In[18]:


history = classifier.fit_generator(training_set,
                                   steps_per_epoch=3,
                                   epochs=2,
                                   # callbacks=callbacks,
                                   validation_data=val_set,
                                   validation_steps=2
                                   )

# In[]
# print(history.history.keys())
# classifier.save("models\ind_model_c.h5")

# In[18]:

predict_data_gen = ImageDataGenerator(rescale=1./255)
prediction_set = predict_data_gen.flow_from_directory(test_img_dir,
                                                      target_size=image_shape,
                                                      batch_size=batch_size,
                                                      class_mode=None,
                                                      shuffle=False)

prediction_set.reset()

pred = classifier.predict_generator(prediction_set, verbose=1)

# print("Predictions", pred)
predicted_class_indices = np.argmax(pred, axis=1)


# print("Predicted class", predicted_class_indices)

print("Training class indices", training_set.class_indices)

labels = (training_set.class_indices)
print(labels.items())
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = prediction_set.filenames
results = pd.DataFrame({"Filename": filenames,
                        "Predictions": predictions,
                        "Class indices": predicted_class_indices})

print(results)

# In[]:


# In[]:
# sum1 = 0
# for x in pred:
#     print(np.sum(x))
#     print(np.max(x))
#     print((np.max(x) / np.sum(x))*100)


# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()


# In[44]:


# img_path = "testset/G/1_080.jpg"

# img = imagepre.load_img(img_path, target_size=(28, 28))
# img_tensor = imagepre.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.

# plt.imshow(img_tensor[0])
# plt.show()

# print(img_tensor.shape)


# In[45]:


# x = imagepre.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# images = np.vstack([x])
# classes = classifier.predict_classes(images, batch_size=10)
# print("Predicted class is:", classes)
