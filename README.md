# LaplaceNet
This repository contains the code for the paper https://arxiv.org/abs/2106.04527
> LaplaceNet: A Hybrid Energy-Neural Model for Deep Semi-Supervised Classification 

This code follows from prior work by https://github.com/CuriousAI/mean-teacher/tree/master/pytorch and https://github.com/ahmetius/LP-DeepSSL and we give our deep thanks to these researchers. 


# Using this repository
Download this repository into some folder, extract the data files, set up the environment and then you are good to go.

## Data Extraction

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
We use what ahmetius's did and you can download the train and test tars from  http://ptak.felk.cvut.cz/personal/toliageo/share/lpdeep/ and extract them in the following directory 
```
>> ./data-local/images/miniimagenet/
```

## Setting Up Environment 
Requirements 
- faiss gpu 1.7.1
- pytorch 1.8.1
- cuda 10.2
- scipy 1.6.2
- tqdm 4.61.0

From a clean conda enviroment you can perform the following commands to get a suitable enviroment
- conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
- conda install -c pytorch faiss-gpu 
- conda install -c anaconda scipy 
- conda install -c conda-forge tqdm 
- pip install torchnet 

## Running the program
To recreate the cifar-10 results from the main paper for 4k labels (for any label split)  you can run 

> python main.py --dataset cifar10 --model wrn-28-2 --num-labeled 4000 --alpha 1.0 --lr 0.03 --labeled-batch-size 48 --batch-size 300 --aug-num 3 --label-split 12 --progress True

To recreate the cifar-100 results from the main paper for 10k labels (for any label split)  you can run 

> python main.py --dataset cifar100 --model wrn-28-2 --num-labeled 10000 --alpha 0.5 --lr 0.03 --labeled-batch-size 50 --aug-num 3 --label-split 12 --progress True

To recreate the miniimagenet results from the main paper for 4k labels (for any label split)  you can run 

> python main.py --dataset miniimagenet --model resnet18 --num-labeled 4000 --alpha 0.5 --lr 0.1 --labeled-batch-size 50 --aug-num 3 --label-split 12 --progress True


