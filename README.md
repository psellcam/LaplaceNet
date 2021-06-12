# LaplaceNet
This repository contains the code for the paper https://arxiv.org/abs/2106.04527
> LaplaceNet: A Hybrid Energy-Neural Model for Deep Semi-Supervised Classification 

Please cite our work if you use this code in your paper

This code follows from prior work by https://github.com/CuriousAI/mean-teacher/tree/master/pytorch and https://github.com/ahmetius/LP-DeepSSL and we give our deep thanks to these researchers. 


# Using this repository
This repository contains all the information you would need to recreate the experiments from our paper and use our code. After downloading and extracting this repository you need to extract the data files, set up a suitable environment and then you can run the code. We give a guide on doing so below

## Data Extraction

Run these commands to extract the data for CIFAR-10/100 , starting from the base path you installed the repo to.

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
We use ahmetius's approach meaning thatyou can download the train and test tars from  http://ptak.felk.cvut.cz/personal/toliageo/share/lpdeep/ and extract them in the following directory 
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
- numpy

From a clean conda enviroment you can perform the following commands to get a suitable enviroment
- conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
- conda install -c pytorch faiss-gpu 
- conda install -c anaconda scipy 
- conda install -c conda-forge tqdm 
- pip install torchnet 

Note that faiss-gpu has some compatibaility issues with certain versions of pytorch but the combination above is certified to work

## Running the program
To recreate the cifar-10 results from the main paper for 4k labels (for any label split)  you can run 

> python main.py --dataset cifar10 --model wrn-28-8 --num-labeled 4000 --alpha 1.0 --lr 0.03 --labeled-batch-size 48 --batch-size 300 --aug-num 3 --label-split 12 --progress True

To recreate the cifar-100 results from the main paper for 10k labels (for any label split)  you can run 

> python main.py --dataset cifar100 --model wrn-28-8 --num-labeled 10000 --alpha 0.5 --lr 0.03 --labeled-batch-size 50 --aug-num 3 --label-split 12 --progress True

To recreate the miniimagenet results from the main paper for 4k labels (for any label split)  you can run 

> python main.py --dataset miniimagenet --model resnet18 --num-labeled 4000 --alpha 0.5 --lr 0.1 --labeled-batch-size 50 --aug-num 3 --label-split 12 --progress True

Increasing --aug-num should give better performance at a cost to computational performance.

## Command line arguments

The documentation for the command line arguments can be found in config/cli.py. Here we give some extra information on the most important ones.

- --dataset : Current available options are cifar10, cifar100 and miniimagenet. If you want to add your own dataset you would need to add the relevant images and label information to the data-local folder in the same format as the other datasets, then you will need to update the config/datasets.py folder to include your new dataset and then finally change the load_args function in the helpers.py. You may potentially need to change the --train-subdir and --eval-subdir options as well to make sure you are pointing to the right folders. 

- --model : Current avaiable options are resnet18, resnet50, wrn-28-2, wrn-28-8 and a 13-cnn. If you want to add your own custom model you would need to add the code to the models subfolder, update the init and then add your model as an option to the create-model function in helpers.py

- --label-split : For a fair comparison we use the same label splits as past works, these are numbered from 0 to 20 for each differing label amount. This label split is then used in the create_data_loaders_simple function in helpers.py where it is sent to the custom dataset class db_semisuper. For some data sets it may make more sense to generate such splits on the fly by changing the relabel dataset function of the db_semisuper class found in the lp folder but to steal a phrase "I will leave this as an exercise to the reader".

- --aug-num : This sets the number of augmentation samples per point as dicussed in the main paper. We fill a value of 3 or 5 is best in most cases.


There are some graph based parameters which we do not offer as cli arguments, these make be changed directly but I don't recommned doing so unless you have a good reason in mind. If you want to try another graph based approach or any propogator then you would need to rewrite the one_iter_true function in db_semisuper.py and replace it with whatever you liked.




## Maintenance

I will try my best to keep this github up to date. If you find a bug or want to make a comment please feel free to do so and I will try my best to resolve your problem quickly.
Additionally I aim, if my PhD time allows, to add to this github with distributed training etc. 
