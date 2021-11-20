#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Input, MaxPooling2D, Flatten, add, Activation
from keras.layers.normalization import BatchNormalization
from keras.metrics import binary_accuracy

import warnings
warnings.filterwarnings("ignore")


def resBlock(ipt, filters, increDimen=False):
    """
    Residual blocks for extracting more deep, effective and distinguishable features.
    """
    res = ipt

    if increDimen:
        ipt = MaxPooling2D(pool_size=(2, 2), padding="same")(ipt)
        res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same")(res)

    out = BatchNormalization()(ipt)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = add([res, out])

    return out


def MGF6mARice():
    """
    MGF6mARice model for rice 6mA sites prediction.
    """
    num_nodes = 11      # # maxNumAtoms of G == 11
    num_features = 17   # # according to the number of features of each atom, which features used can be self-define
    seqLength = 41

    dropout1 = 0.285293526161375
    dropout2 = 0.746951714170157

    # Input features about molecular graph features
    features = Input(shape=(seqLength, num_nodes, num_features))

    # Convolutional layer at the beginning for preliminary feature extraction
    conv2dLayer = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", name='conv2d')(features)

    # Residual blocks
    resBlockRes1 = resBlock(conv2dLayer, 32)
    resBlockRes2 = resBlock(resBlockRes1, 64, increDimen=True)

    flattenLayer = Flatten()(resBlockRes2)

    # MLP to build a prediction
    dense1Layer = Dense(256, activation='relu', name='dense1')(flattenLayer)
    dropout1Layer = Dropout(rate=dropout1, name='dropout1')(dense1Layer)
    dense2Layer = Dense(64, kernel_initializer='glorot_normal', activation='relu', name='dense2')(dropout1Layer)
    dropout2Layer = Dropout(rate=dropout2, name='dropout2')(dense2Layer)
    dense3layer = Dense(32, activation='relu', name='dense3')(dropout2Layer)

    # The probability of samples being a 6mA site
    pred = Dense(1, activation='sigmoid', name='dense4')(dense3layer)

    model = Model(inputs=features, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[binary_accuracy])

    print(model.summary())

    return model
