#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, \
    roc_curve, roc_auc_score, auc, precision_recall_curve
import matplotlib.pyplot as plt
from scipy import interp


def calculateScore(X, y, model, OutputFile):
    """
    :param X         : data
    :param y         : labels
    :param model     : model
    :param OutputFile: output file about predict results
    :return          : metrics
    """
    score = model.evaluate(X, y)
    pred_y = model.predict(X)

    with open(OutputFile, 'w') as fOUT:
        for index in range(len(y)):
            fOUT.write(str(y[index])+'\t'+str(pred_y[index][0])+'\n')

    accuracy = score[1]

    tempLabel = [(0 if i < 0.5 else 1) for i in pred_y]
    confusion = confusion_matrix(y, tempLabel)

    TN, FP, FN, TP = confusion.ravel()
    sensitivity = recall_score(y, tempLabel)
    specificity = TN / float(TN + FP)
    MCC = matthews_corrcoef(y, tempLabel)
    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)

    pred_y = pred_y.reshape((-1,))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)

    precisionPR, recallPR, _ = precision_recall_curve(y, pred_y)
    aupr = auc(recallPR, precisionPR)

    return {'sn': sensitivity, 'sp': specificity, 'acc': accuracy, 'MCC': MCC, 'AUC': ROCArea, 'precision': precision,
            'F1': F1Score, 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'AUPR': aupr, 'precisionPR': precisionPR,
            'recallPR': recallPR, 'y_real': y, 'y_pred': pred_y}


def analyze(temp, OutputDir):
    """
    Metrics and plot.
    """
    plt.cla()
    plt.style.use("ggplot")

    trainning_result, validation_result, testing_result = temp

    # The performance output file about training, validation, and test set
    file = open(OutputDir + '/performance.txt', 'w')
    index = 0
    for x in [trainning_result, validation_result, testing_result]:
        title = ''
        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'
        index += 1
        file.write(title + 'results\n')

        for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'AUPR', 'precision', 'F1']:
            total = []
            for val in x:
                total.append(val[j])
            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total)) + '\n')
        file.write('\n\n______________________________\n')
    file.close()

    # Plot ROC about training, validation, and test set
    indexROC = 0
    for x in [trainning_result, validation_result, testing_result]:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))
            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        titleROC = ''
        if indexROC == 0:
            titleROC = 'training_'
        if indexROC == 1:
            titleROC = 'validation_'
        if indexROC == 2:
            titleROC = 'testing_'
        plt.savefig(OutputDir + '/' + titleROC + 'ROC.png')
        plt.close('all')

        indexROC += 1

    # Plot PR curve about training, validation, and test set
    indexPR = 0
    for item in [trainning_result, validation_result, testing_result]:
        y_realAll = []
        y_predAll = []
        i = 0
        for val in item:
            precisionPR = val['precisionPR']
            recallPR = val['recallPR']
            aupr = val['AUPR']
            plt.plot(recallPR, precisionPR, lw=1, alpha=0.3, label='PR fold %d (AUPR = %0.2f)' % (i + 1, aupr))

            y_realAll.append(val['y_real'])
            y_predAll.append(val['y_pred'])
            i += 1

        y_realAll = np.concatenate(y_realAll)
        y_predAll = np.concatenate(y_predAll)

        precisionPRAll, recallPRAll, _ = precision_recall_curve(y_realAll, y_predAll)
        auprAll = auc(recallPRAll, precisionPRAll)

        plt.plot(recallPRAll, precisionPRAll, color='b', label=r'Precision-Recall (AUPR = %0.2f)' % (auprAll),
                 lw=1, alpha=.8)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        titlePR = ''
        if indexPR == 0:
            titlePR = 'training_'
        if indexPR == 1:
            titlePR = 'validation_'
        if indexPR == 2:
            titlePR = 'testing_'
        plt.savefig(OutputDir + '/' + titlePR + 'PR.png')
        plt.close('all')

        indexPR += 1
