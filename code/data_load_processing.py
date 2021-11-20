#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random
import pandas as pd
import numpy as np
from features import convert_to_graph


def chunkIt(seq, num):
    """
    Divide the data based on k-folds.
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def dataProcessing(path):
    """
    Read the data, and then encode the DNA molecular graph feature.
    """
    data = pd.read_csv(path)
    X = []
    for line in data['data']:
        seq = line.strip('\n')
        graphFeatures = convert_to_graph(seq)
        X.append(graphFeatures)
    X = np.asarray(X)                            # # (bs, 41, 11 ,17)
    y = np.array(data['label'], dtype=np.int32)  # # (bs, )

    return X, y


def prepareData(PositiveCSV, NegativeCSV):
    """
    :param PositiveCSV: the positive samples of input file with comma-separated values.
    :param NegativeCSV: the negative samples of input file with comma-separated values.
    :return           : DNA molecular graph features of positive and negative samples and their corresponding labels.
    """
    Positive_X, Positive_y = dataProcessing(PositiveCSV)
    Negitive_X, Negitive_y = dataProcessing(NegativeCSV)

    return Positive_X, Positive_y, Negitive_X, Negitive_y


def shuffleData(X, y):
    """
    :param X: data
    :param y: labels
    :return : data and labels after shuffle
    """
    index = [i for i in range(len(X))]
    random.seed(0)
    random.shuffle(index)
    X = X[index]
    y = y[index]

    return X, y
