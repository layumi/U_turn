<h1 align="center"> :see_no_evil: U-Turn :hear_no_evil: </h1>
<h2 align="center"> Attack your retrieval model via Query! They are not robust as you expected!  </h2>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

One simple code to cheat your retrieval model via **Modifying Query ONLY** (based on [pytorch](https://pytorch.org)) accepted by IJCV.
Pre-print version is at https://arxiv.org/abs/1809.02681.  

The main idea underpinning our method is simple yet effective, making the query feature to conduct a U-turn :arrow_right_hook:.

![](https://github.com/layumi/A_reID/blob/master/method.png)

## Table of contents
* [Re-ID Attacking](#re-id-attacking)
* [Image Retrieval Attacking](#image-retrieval-attacking)
* [Cifar Attacking](#mnist-attacking)


## Re-ID Attacking
### 1.1 Preparing your reID models. 
Please check the step-by-step tutorial in https://github.com/layumi/Person_reID_baseline_pytorch

### 1.2 Attacking Market-1501 
Try four attack methods with one line. Please change the path before run it.
```bash
python experiment.py
```

## Image Retrieval Attacking
### 2.1 Download the pre-trained model on Oxford and Paris
We attach the training code, which is based on the excellent code in TPAMI 2018. 
https://github.com/layumi/Oxford-Paris-Attack

### 2.2 Attacking the Oxford and Paris Dataset 
Our effort is to cheat the TPAMI model. Yes. We succeed. 
https://github.com/layumi/Oxford-Paris-Attack 

### 2.3 Attacking Food-256 and CUB-200-2011 
Please check subfolders.

Food: https://github.com/layumi/U_turn/tree/master/Food

CUB: https://github.com/layumi/U_turn/tree/master/cub

## Cifar Attacking
### 3.1 Cifar (ResNet-Wide)
We attach the training code, which is borrowed from ResNet-Wide (with Random Erasing).

### 3.2 Attacking Cifar
https://github.com/layumi/A_reID/tree/master/cifar 


![](https://github.com/layumi/pytorch-mnist/blob/master/train.jpg)



