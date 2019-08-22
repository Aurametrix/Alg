ort os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datetime import datetime
from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.layers import Reshape, concatenate, LeakyReLU, Lambda
from keras.layers import K, Activation, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import image
from scipy.io import loadmat

def build_encoder():

    """
    Encoder Network
    """
    
    input_layer = Input(shape = (64, 64, 3))
    
    ## 1st Convolutional Block
    enc = Conv2D(filters = 32, kernel_size = 5, strides = 2, padding = 'same')(input_layer)
    enc = LeakyReLU(alpha = 0.2)(enc)
    
    ## 2nd Convolutional Block
    enc = Conv2D(filters = 64, kernel_size = 5, strides = 2, padding = 'same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)
    
    ## 3rd Convolutional Block
    enc = Conv2D(filters = 128, kernel_size = 5, strides = 2, padding = 'same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)
    
    ## 4th Convolutional Block
    enc = Conv2D(filters = 256, kernel_size = 5, strides = 2, padding = 'same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)
    
    
    ## Flatten layer
    enc = Flatten()(enc)
    
    ## 1st Fully Connected Layer
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)
    
    ## 2nd Fully Connected Layer
    enc = Dense(100)(enc)
    
    
    ## Create a model
    model = Model(inputs = [input_layer], outputs = [enc])
    return model
    
    
    def build_generator():

    """
    Generator Network
    """
    
    latent_dims = 100
    num_classes = 6
    
    input_z_noise = Input(shape = (latent_dims, ))
    input_label = Input(shape = (num_classes, ))
    
    x = concatenate([input_z_noise, input_label])
    
    x = Dense(2048, input_dim = latent_dims + num_classes)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256 * 8 * 8)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)
    
    x = Reshape((8, 8, 256))(x)
   
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(filters = 128, kernel_size = 5, padding = 'same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(filters = 64, kernel_size = 5, padding = 'same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(filters = 3, kernel_size = 5, padding = 'same')(x)
    x = Activation('tanh')(x)
    
    model = Model(inputs = [input_z_noise, input_label], outputs = [x])
    return model
    
    
    def build_generator():

    """
    Generator Network
    """
    
    latent_dims = 100
    num_classes = 6
    
    input_z_noise = Input(shape = (latent_dims, ))
    input_label = Input(shape = (num_classes, ))
    
    x = concatenate([input_z_noise, input_label])
    
    x = Dense(2048, input_dim = latent_dims + num_classes)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(256 * 8 * 8)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Dropout(0.2)(x)
    
    x = Reshape((8, 8, 256))(x)
   
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(filters = 128, kernel_size = 5, padding = 'same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(filters = 64, kernel_size = 5, padding = 'same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    
    x = UpSampling2D(size = (2, 2))(x)
    x = Conv2D(filters = 3, kernel_size = 5, padding = 'same')(x)
    x = Activation('tanh')(x)
    
    model = Model(inputs = [input_z_noise, input_label], outputs = [x])
    return model
    
    # Face Recognition Network
    
    def build_fr_model(input_shape):

    resnet_model = InceptionResNetV2(include_top = False, weights = 'imagenet', 
                                     input_shape = input_shape, pooling = 'avg')
    image_input = resnet_model.input
    x = resnet_model.layers[-1].output
    out = Dense(128)(x)
    embedder_model = Model(inputs = [image_input], outputs = [out])
    
    input_layer = Input(shape = input_shape)
    
    x = embedder_model(input_layer)
    output = Lambda(lambda x: K.l2_normalize(x, axis = -1))(x)
    
    model = Model(inputs = [input_layer], outputs = [output])
    return model
    
    
    # 
