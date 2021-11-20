#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project    ：MGF6mARice: prediction of DNA N6-methyladenine sites in rice by exploiting molecular graph feature and residual block.
@Description: 10-fold cross-validation of MGF6mARice.
'''
print(__doc__)

import sys, argparse


def main():

    if not os.path.exists(args.output):
        print("The output path not exist! Create a new folder...\n")
        os.makedirs(args.output)
    if not os.path.exists(args.positive) or not os.path.exists(args.negative):
        print("The input data not exist! Error\n")
        sys.exit()

    funciton(args.positive, args.negative, args.output, args.fold)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Manual to the MGF6mARice')
    parser.add_argument('-p', '--positive', type=str, help='6mA positive data')
    parser.add_argument('-n', '--negative', type=str, help='non-6mA negative data')
    parser.add_argument('-f', '--fold', type=int, help='k-fold cross-validation', default=10)
    parser.add_argument('-o', '--output', type=str, help='output folder')
    args = parser.parse_args()

    from train import *
    main()
