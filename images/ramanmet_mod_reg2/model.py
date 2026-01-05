"""
    RamanNet model definition
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dropout, Dense, BatchNormalization, concatenate, Lambda, LeakyReLU, Attention
import numpy as np

def RamanNetRegression(w_len, n_windows):
    """
    RamanNet model

    Args:
        w_len (int): length of segmented windows
        n_windows (int): number of segmented windows
        n_classes (int, optional): number of output classes

    Returns:
        Tensorflow Keras Model : the RamanNet model
    """

    inps = []                           # input spectrum windows
    features = []                       # extracted features from the windows

    for i in range(n_windows):

        inp = Input(shape=(w_len,))
        inps.append(inp)

        feat = Dense(25)(inp)
        feat = BatchNormalization()(feat)
        feat = LeakyReLU()(feat)
        features.append(feat)


    comb = concatenate(features)            # merging all the features
    comb = Dropout(0.50)(comb)

    top = Dense(512)(comb)
    top = BatchNormalization()(top)
    top = Dense(512)(comb)

    top = BatchNormalization()(top)
    top = LeakyReLU()(top)
    top = Dropout(0.40)(top)


    top = Dense(256)(top)

    emb = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='embedding')(top)      # computed embedding

    top = BatchNormalization()(top)
    top = LeakyReLU()(top)
    top = Dropout(0.25)(top)

    
    #rpet_output = Dense(1, activation='linear', name='rpet')(top)
    temp_output = Dense(1, activation='linear', name='temperature')(top)
    #PCA_output = Dense(1, activation='linear', name='PCA')(top)

    mdl = Model(inputs=inps, outputs={
        "temperature": temp_output,
        # "rpet": rpet_output
        #"PCA": PCA_output
    })  

    return mdl

