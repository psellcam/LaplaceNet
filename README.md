# LaplaceNet
This repository contains the code for the paper https://arxiv.org/abs/2106.04527
> LaplaceNet: A Hybrid Energy-Neural Model for Deep Semi-Supervised Classification 

This code follows from prior work by https://github.com/CuriousAI/mean-teacher/tree/master/pytorch and https://github.com/ahmetius/LP-DeepSSL and we give our deep thanks to these researchers. 


# Using this repository
Download this repository into some folder

## Data Preprocess

### CIFAR-10
```
>> cd data-local/bin
>> ./prepare_cifar10.sh
```

### CIFAR-100
```
>> cd data-local/bin
>> ./prepare_cifar100.sh
```

### Mini-Imagenet
We took the Mini-Imagenet dataset hosted in [this repository](https://github.com/gidariss/FewShotWithoutForgetting) and pre-processed it.
Download [train.tar.gz](http://ptak.felk.cvut.cz/personal/toliageo/share/lpdeep/train.tar.gz) and [test.tar.gz](http://ptak.felk.cvut.cz/personal/toliageo/share/lpdeep/test.tar.gz), and extract them in the following directory:
```
>> ./data-local/images/miniimagenet/
