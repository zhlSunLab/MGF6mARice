# MGF6mARice

## Introduction

```text
MGF6mARice: prediction of DNA N6-methyladenine sites in rice by exploiting molecular graph feature and residual block
```

Inspired by the generation mechanism of 6mA, we construct a new feature, that is DNA molecular graph feature, which is mined and calculated from the chemical molecular structure (SMILES format) of DNA bases. In addition, considering that the residual block is widely used in bioinformatics but not used in rice 6mA prediction till now, it is also utilized in this research, due to its ability to extract higher-level and easier-to-distinct features. Therefore, we propose a novel deep learning method based on DNA molecular graph feature and residual block for rice 6mA sites prediction, abbreviated as MGF6mARice. Experiments have shown that the molecular graph feature and residual block can promote the performance of MGF6mARice in 6mA prediction. We hope that MGF6mARice is an effective tool for researchers to analyze rice 6mA sites, and the DNA molecular graph feature can also be transferred to other methylation type prediction, in terms of the nucleic acid sequence.

## Data

#### SMILES strings for DNA bases

```shell
cat ./data/basesSMILES/ATGC.dict
```

#### Benchmark datasets

##### Rice_Chen dataset

```shell
cd ./data/6mA_data/Rice_Chen/
```

**Positive.txt** is the input file of positive samples. **Negative.txt** is the input file of negative samples. **./train_val_test_10CV_data/*** is the folder including training, validation, and test set of each fold.

##### Rice_Lv dataset

```shell
cd ./data/6mA_data/Rice_Lv/
```

**pos.txt** is the input file of positive samples. **neg.txt** is the input file of negative samples. **./train_val_test_10CV_data/*** is the folder including training, validation, and test set of each fold.

#### Imbalanced datasets

##### From Rice_Chen dataset

```shell
cd ./data/ImbalancedData/Rice_Chen/
```

* **Positive_1_5.txt** is constructed by random selection from **Positive.txt** via 1:5 of selection ratio.
* **Positive_1_10.txt** is constructed by random selection from **Positive.txt** via 1:10 of selection ratio.
* **Positive_1_20.txt** is constructed by random selection from **Positive.txt** via 1:20 of selection ratio.

##### From Rice_Lv dataset

```shell
cd ./data/ImbalancedData/Rice_Lv/
```

* **pos_1_5.txt** is constructed by random selection from **pos.txt** via 1:5 of selection ratio.
* **pos_1_10.txt** is constructed by random selection from **pos.txt** via 1:10 of selection ratio.
* **pos_1_20.txt** is constructed by random selection from **pos.txt** via 1:20 of selection ratio.

#### Independent datasets

##### Same species independent dataset

```shell
cd ./data/IndependentData/NIP_10000/
```

##### Cross species independent datasets

```shell
cd ./data/IndependentData/A.thaliana/
cd ./data/IndependentData/D.melanogaster/
cd ./data/IndependentData/R.chinensis/
```

## Requirement

The code has been tested running under Python 3.5.6. 

The required packages are as follows:

* cudatoolkit=10.0
* cudnn=7.6.4
* tensorflow-gpu==1.13.2
* keras==2.1.5
* numpy==1.16.4
* pandas==0.20.3
* scikit-learn==0.22.2.post1
* rdkit==2017.03.1

## Usage

```shell
git clone https://github.com/zhlSunLab/MGF6mARice
cd ./MGF6mARice/

# show the help message
python ./code/main.py -h

# example for train
python ./code/main.py -p ./data/6mA_data/Rice_Chen/Positive.txt -n ./data/6mA_data/Rice_Chen/Negative.txt -f 10 -o ./result/Rice_Chen_10CV/
```

## Contact

Please feel free to contact us if you need any help (E-mail: mengyaliu1003@foxmail.com).

## Citation
```reference
@article{liu2022mgf6marice,
  title={MGF6mARice: prediction of DNA N6-methyladenine sites in rice by exploiting molecular graph feature and residual block},
  author={Liu, Mengya and Sun, Zhan-Li and Zeng, Zhigang and Lam, Kin-Man},
  journal={Briefings in Bioinformatics},
  volume={23},
  number={3},
  pages={bbac082},
  year={2022},
  publisher={Oxford University Press}
}
```
