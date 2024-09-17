from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import os

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from PIL import Image
import glob

#from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

npz_comp = np.load('./af1_test.npz')
npz_comp1 = np.load('./af1_train.npz')
test_label = npz_comp['l']
test_data = npz_comp['d']
train_label = npz_comp1['l']
train_data = npz_comp1['d']

le=len(train_data)
le2=len(test_data)

train_data = train_data.reshape((le, 256, 256,1))
train_label = train_label.reshape((le, 1))
test_data = test_data.reshape((le2, 256, 256,1))
test_label = test_label.reshape((le2, 1))

for num in range(5):
    tf.random.set_seed(0)
	#os.makedirs("log%d"%num, exist_ok=True)
    model = models.Sequential()
    model.add(layers.Conv2D(8, (3,3), activation='relu', input_shape=(256,256,1)))
    model.add(layers.Conv2D(8, (3,3), activation='relu'))
    model.add(layers.Conv2D(8, (3,3), activation='relu'))
    model.add(layers.Conv2D(8, (3,3), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.Conv2D(16, (3,3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    #model.summary()

    model.compile(optimizer='adam', 
              #loss='categorical_crossentropy', 
              loss='binary_crossentropy', 
			  metrics=['acc'])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
        filepath="log%d/mymodel_{epoch}"%num,
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        #monitor="loss",
        monitor="val_acc",
        verbose=2,
        )
    ]
    model.fit(train_data, train_label, epochs=500, batch_size=100, callbacks=callbacks, validation_split=0.1, verbose=2)
    test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)
    model.save('log%d/model75'%num)
