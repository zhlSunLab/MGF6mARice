#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib as mpl
mpl.use('Agg')

from data_load_processing import *
from model import MGF6mARice
from metrics_plot import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gc


def funciton(PositiveCSV, NegativeCSV, OutputDir, folds):
    """
    :param PositiveCSV: the positive samples of input file with comma-separated values.
    :param NegativeCSV: the negative samples of input file with comma-separated values.
    :param OutputDir  : directory of output.
    :param folds      : k-fold(s) cross-validation.
    :return           : results, performance and plots of k-fold(s) cross-validation.
    """
    Positive_X, Positive_y, Negative_X, Negative_y = prepareData(PositiveCSV, NegativeCSV)

    random.seed(7)
    random.shuffle(Positive_X)
    random.shuffle(Negative_X)

    Positive_X_Slices = chunkIt(Positive_X, folds)
    Positive_y_Slices = chunkIt(Positive_y, folds)
    Negative_X_Slices = chunkIt(Negative_X, folds)
    Negative_y_Slices = chunkIt(Negative_y, folds)

    del Positive_X, Negative_X, Positive_y, Negative_y
    gc.collect()

    trainning_result = []
    validation_result = []
    testing_result = []

    for test_index in range(folds):

        test_X = np.concatenate((Positive_X_Slices[test_index], Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index], Negative_y_Slices[test_index]))

        validation_index = (test_index + 1) % folds
        valid_X = np.concatenate((Positive_X_Slices[validation_index], Negative_X_Slices[validation_index]))
        valid_y = np.concatenate((Positive_y_Slices[validation_index], Negative_y_Slices[validation_index]))

        start = 0
        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val
                break
        train_X = np.concatenate((Positive_X_Slices[start], Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start], Negative_y_Slices[start]))
        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i], Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i], Negative_y_Slices[i]))
                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))

        test_X, test_y = shuffleData(test_X, test_y)
        valid_X, valid_y = shuffleData(valid_X, valid_y)
        train_X, train_y = shuffleData(train_X, train_y)

        model = MGF6mARice()

        # early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=30)
        model_check = ModelCheckpoint(filepath=OutputDir + "/model" + str(test_index + 1) + ".h5",
                                      monitor='val_binary_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20)

        model.fit(train_X, train_y, batch_size=32, epochs=100, validation_data=(valid_X, valid_y),
                  callbacks=[model_check, reduct_L_rate])

        trainning_result.append(calculateScore(train_X, train_y, model,
                                               OutputDir + "/trainy_predy_" + str(test_index+1) + ".txt"))
        validation_result.append(calculateScore(valid_X, valid_y, model,
                                                OutputDir + "/validy_predy_" + str(test_index+1) + ".txt"))
        testing_result.append(calculateScore(test_X, test_y, model,
                                             OutputDir + "/testy_predy_" + str(test_index+1) + ".txt"))

        del model, train_X, valid_X, test_X, train_y, valid_y, test_y
        gc.collect()

    temp_dict = (trainning_result, validation_result, testing_result)
    analyze(temp_dict, OutputDir)

    del trainning_result, validation_result, testing_result
    gc.collect()
