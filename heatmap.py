from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import argparse
import sys
from tensorflow.keras import backend as K
from tensorflow.keras.utils import array_to_img, img_to_array, load_img

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import os

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from PIL import Image
import glob


def grad_cam(input_data, layer_name):
    
    g = tf.Graph()
    with g.as_default():
        
        input_model = tf.keras.models.load_model('./')
        model = input_model
        model.summary()

        y_c = model.output[0, 0]
        
        conv_output = model.get_layer(layer_name).output
        grads = K.gradients(y_c, conv_output)[0]

        # Get outputs and grads
        gradient_function = K.function([model.input], [conv_output, grads])
        output, grads_val = gradient_function([input_data])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis=(0, 1)) # Passing through GlobalAveragePooling

        cam = np.dot(output, weights) # multiply
        cam = np.maximum(cam, 0)      # Passing through ReLU
        cam /= np.max(cam)            # scale 0 to 1.0

        grad_CAM_map = cv2.resize(cam, (256, 256), cv2.INTER_LINEAR)
        jetcam = cv2.applyColorMap(np.uint8(255 * grad_CAM_map), cv2.COLORMAP_JET) 
        return jetcam 


def grad_cam_plus(input_data, layer_name):
    """Grad-CAM++ function"""
    g = tf.Graph()
    with g.as_default():

        input_model = tf.keras.models.load_model('./')
        model = input_model
        model.summary()

        y_c = model.output[0,0]

        conv_output = model.get_layer(layer_name).output
        grads = K.gradients(y_c, conv_output)[0]

        first = K.exp(y_c) * grads
        second = K.exp(y_c) * grads * grads
        third = K.exp(y_c) * grads * grads * grads

        gradient_function = K.function([model.input], [y_c, first, second, third, conv_output, grads])
        y_c, conv_first_grad, conv_second_grad, conv_third_grad, conv_output, grads_val = gradient_function([input_data])
        global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape((1, 1, conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num / alpha_denom

        weights = np.maximum(conv_first_grad[0], 0.0)
        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
        alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))
        deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)

        cam = np.sum(deep_linearization_weights * conv_output[0], axis=2)
        cam = np.maximum(cam, 0) # Passing through ReLU
        cam /= np.max(cam)       # scale 0 to 1.0
        
        grad_CAM_map = cv2.resize(cam, (256, 256), cv2.INTER_LINEAR)
        jetcam = cv2.applyColorMap(np.uint8(255 * grad_CAM_map), cv2.COLORMAP_JET) 
        return jetcam 

#患者ナンバーの選択
number = 129 #SR:0〜113  AF:114〜  #16,103,121,133,136

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

npz_comp = np.load('./af4_test.npz')
test_label = npz_comp['l']
test_data = npz_comp['d']
le2=len(test_data)
test_data = test_data.reshape((le2, 256, 256, 1))
test_label = test_label.reshape((le2, 1))

tf.random.set_seed(0)

input_data = []
input_data.append(test_data[number])
input_data = np.array(input_data)


"""
rslt = model.predict(test_data, batch_size=None, verbose=2, steps=None)
rslt2 = np.round(rslt)

for z in range(len(rslt2)):
	pre = int(rslt2[z][0])
	sei = int(test_label[z][0])
	if (pre != sei):
		print(z)	
"""



for conv_number in range(48,64):
	
	if (conv_number == 48):
		layer_name = 'conv2d_48'
	else:
		layer_name = 'conv2d_%d'%conv_number
	
	#ヒートマップ作成
	#zu1 = grad_cam(input_data, layer_name)
	zu1 = grad_cam_plus(input_data, layer_name)

	#画像の復元
	x = test_data[number]
	x = x.reshape((256, 256))
	for i in range(256):
		for j in range(256):
			if (int(x[i][j])==0):
				x[i][j] = 0
			else:
				x[i][j] = 255
	x = x.reshape((256, 256, 1))
	cv2.imwrite('HeatMap/original.png',x)
	
	#ヒートマップ画像の生成
	jetcam1 = (np.float32(zu1)+x/2)
	cv2.imwrite('HeatMap/%s.png'%layer_name,jetcam1)

