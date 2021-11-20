#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from rdkit import Chem

import warnings
warnings.filterwarnings("ignore")


def basesDict():
    """
    Obtain SMILES strings of A, T, G, C molecule.
    """
    baseDict = {}
    with open('./data/basesSMILES/ATGC.dict', 'r') as f:
        for line in f:
            line = line.strip().split(',')
            baseDict[line[0]] = line[1]

    return baseDict


def atom_feature(atom):
    """
    Features of atoms in base molecule.
    """
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O']) +                      # # whether there is an atom in the molecule
                    one_of_k_encoding(atom.GetDegree(), [1, 2, 3]) +              # # the degree of the atom in the molecule
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3]) +       # # the total number of hydrogens on the atom
                    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3]) +  # # the implicit valence
                    [atom.GetIsAromatic()] +                                      # # whether the atom is aromatic
                    get_ring_info(atom))                                          # # the size of atomic ring
                                                                                  # # feature dimensions are 17


def one_of_k_encoding(x, allowable_set):

    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):

    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def get_ring_info(atom):

    ring_info_feature = []
    for i in range(5, 7):                                                   # # base A: 5,6
        if atom.IsInRingSize(i):
            ring_info_feature.append(1)
        else:
            ring_info_feature.append(0)

    return ring_info_feature


def norm_Adj(adjacency):
    """
    Obtain the normalized adjacency matrix.
    """
    I = np.array(np.eye(adjacency.shape[0]))                                # # identity matrix
    adj_hat = adjacency + I

    # D^(-1/2) * (A + I) * D^(-1/2)
    D_hat = np.diag(np.power(np.array(adj_hat.sum(1)), -0.5).flatten(), 0)  # # degree matrix
    adj_Norm = adj_hat.dot(D_hat).transpose().dot(D_hat)                    # # normalized adjacency matrix

    return adj_Norm


def norm_fea(features):
    """
    Obtain the normalized node feature matrix.
    """
    norm_fea = features / features.sum(1).reshape(-1, 1)                    # # normalized node feature matrix

    return norm_fea


def convert_to_graph(seq):
    """
    Obtain the molecular graph features of one sequence.
    """
    baseDict = basesDict()
    maxNumAtoms = 11            # # base G has the maxNumAtoms

    # Molecules of bases from one sequence
    graphFeaturesOneSeq = []
    seqSMILES = [baseDict[b] for b in seq]
    for baseSMILES in seqSMILES:
        DNA_mol = Chem.MolFromSmiles(baseSMILES)

        # Adjacency matrix
        AdjTmp = Chem.GetAdjacencyMatrix(DNA_mol)
        AdjNorm = norm_Adj(AdjTmp)

        # Node feature matrix (features of node (atom))
        if AdjNorm.shape[0] <= maxNumAtoms:

            # Preprocessing of feature
            graphFeature = np.zeros((maxNumAtoms, 17))
            nodeFeatureTmp = []
            for atom in DNA_mol.GetAtoms():
                nodeFeatureTmp.append(atom_feature(atom))
            nodeFeatureNorm = norm_fea(np.asarray(nodeFeatureTmp))

            # Molecular graph feature for one base
            graphFeature[0:len(nodeFeatureTmp), 0:17] = np.dot(AdjNorm.T, nodeFeatureNorm)

            # Append the molecualr graph features for bases in order
            graphFeaturesOneSeq.append(graphFeature)

    # Molecular graph features for one sequence
    graphFeaturesOneSeq = np.asarray(graphFeaturesOneSeq, dtype=np.float32)

    return graphFeaturesOneSeq  # # (41, 11, 17)
