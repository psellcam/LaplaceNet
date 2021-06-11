import torchvision.transforms as transforms
from .augmentations import RandAugment
from .utils import export
import os

@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    
    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])


    myhost = os.uname()[1]
    data_dir = 'data-local/images/cifar/cifar10/by-image'

    print("Using CIFAR-10 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }


@export
def cifar100():
    channel_stats = dict(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675,  0.2565,  0.2761]) # should we use different stats - do this
    
    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])


    myhost = os.uname()[1]
    data_dir = 'data-local/images/cifar/cifar100/by-image'

    print("Using CIFAR-100 from", data_dir)

    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }



@export
def miniimagenet():
    mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]

    channel_stats = dict(mean=mean_pix, std=std_pix) 

    weak_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(84, padding=8,padding_mode="reflect"),
        RandAugment(1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    
    strong_transformation = transforms.Compose([        
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(84, padding=8,padding_mode="reflect"),
        RandAugment(2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = 'data-local/images/miniimagenet'
    

    print("Using mini-imagenet from", data_dir)


    return {
        'weak_transformation': weak_transformation,
        'strong_transformation': strong_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }
