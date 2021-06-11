import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from config.utils import *
import lp.db_semisuper as db_semisuper
import lp.db_eval as db_eval
from models import *
import itertools
import torch.backends.cudnn as cudnn
import torchvision

class StreamBatchSampler(Sampler):

    def __init__(self, primary_indices, batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size

    def __iter__(self):
        primary_iter = iterate_eternally(self.primary_indices)
        return (primary_batch  for (primary_batch)
            in  grouper(primary_iter, self.primary_batch_size)
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
    
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())    
    
def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)         





def create_data_loaders_simple(weak_transformation,strong_transformation,
                        eval_transformation,
                        datadir,
                        args):
    
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    with open(args.labels) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines()) 

    dataset =  db_semisuper.DBSS(traindir, labels , False , args.aug_num , eval_transformation,weak_transformation,strong_transformation)
    
    
    sampler = SubsetRandomSampler(dataset.labeled_idx)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset,batch_sampler=batch_sampler,num_workers=args.workers,pin_memory=True)


    train_loader_noshuff = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,  
        pin_memory=True,
        drop_last=False)
    
    eval_dataset = db_eval.DBE(evaldir, False, eval_transformation)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    
    batch_sampler_l = StreamBatchSampler(dataset.labeled_idx, batch_size=args.labeled_batch_size)
    batch_sampler_u = BatchSampler(SubsetRandomSampler(dataset.unlabeled_idx), batch_size=args.batch_size - args.labeled_batch_size, drop_last=True)

    train_loader_l = DataLoader(dataset, batch_sampler=batch_sampler_l,
                                               num_workers=args.workers,
                                               pin_memory=True)

    train_loader_u = DataLoader(dataset, batch_sampler=batch_sampler_u,
                                               num_workers=args.workers,
                                               pin_memory=True)
    

  
    return train_loader, eval_loader, train_loader_noshuff , train_loader_l , train_loader_u , dataset



#### Create Model
def create_model(num_classes,args):
    model_choice = args.model
    
    if model_choice == "resnet18":
        model = resnet18(num_classes)
        
    elif model_choice == "resnet50":
        model = resnet50(num_classes)
        
    elif model_choice == "wrn-28-2":
        model = build_wideresnet(28,2,0,num_classes)
    
    elif model_choice == "wrn-28-8":
        model = build_wideresnet(28,8,0,num_classes)
        
    elif model_choice == "cifarcnn":
        model = cifar_cnn(num_classes)

    model = nn.DataParallel(model)
    model.to(args.device)
    cudnn.benchmark = True    
    return model


def hellinger(p,q):
    return np.sqrt(np.sum((np.sqrt(p)-np.sqrt(q))**2))/np.sqrt(2)

def mixup_data(x_1 , index , lam):
    mixed_x_1 = lam * x_1 + (1 - lam) * x_1[index, :]        
    return mixed_x_1 

def mixup_criterion(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()      
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_sup(train_loader, model, optimizer, epoch, global_step, args, ema_model = None):
    # switch to train mode
    model.train()
    for i, (aug_images , target) in enumerate(train_loader):      
        target = target.to(args.device)          
        #Create the mix
        alpha = args.alpha     
        index = torch.randperm(args.batch_size,device=args.device)        
        lam = np.random.beta(alpha, alpha)
        target_a, target_b = target, target[index]   
        
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
        
        # Loop over the batches 
        count = 0
        for batch in aug_images:
            batch = batch.to(args.device)
            m_batch = mixup_data(batch,index,lam)
            class_logit , _  = model(m_batch)
            if count == 0:
                loss_sum =  mixup_criterion(class_logit.double(), target_a , target_b , lam).mean()
            else:
                loss_sum += mixup_criterion(class_logit.double(), target_a , target_b , lam).mean()

            count += 1        

        loss = loss_sum / (args.aug_num)
        loss.backward()
        optimizer.step()	    
        global_step += 1
    return global_step

def train_semi(train_loader_l, train_loader_u , model, optimizer, epoch, global_step, args, ema_model = None):

    # switch to train mode
    model.train()
    lr_length = len(train_loader_u)
    train_loader_l = iter(train_loader_l)
    
    if args.progress == True:
        from tqdm import tqdm         
        from torchnet import meter
        tk0 =  tqdm(train_loader_u,desc="Semi Supervised Learning Epoch " + str(epoch) + "/" +str(args.epochs),unit="batch")
        loss_meter = meter.AverageValueMeter()
    else: 
        tk0 = train_loader_u
        
    
    for i, (aug_images_u,target_u) in enumerate(tk0):            
        aug_images_l,target_l = next(train_loader_l)
        
        target_l = target_l.to(args.device)
        target_u = target_u.to(args.device)        
        target = torch.cat((target_l,target_u),0)    
        
        #Create the mix
        alpha = args.alpha     
        index = torch.randperm(args.batch_size,device=args.device)
        lam = np.random.beta(alpha, alpha)   
        target_a, target_b = target, target[index]            

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, i, lr_length, args)        
        
        count = 0
        for batch_l , batch_u in zip(aug_images_l ,aug_images_u):
            batch_l = batch_l.to(args.device)
            batch_u = batch_u.to(args.device)
            batch = torch.cat((batch_l,batch_u),0) 
            m_batch = mixup_data(batch,index,lam)            
            class_logit , _  = model(m_batch)

            if count == 0:
                loss_sum =  mixup_criterion(class_logit.double() , target_a , target_b , lam).mean()
            else:
                loss_sum += mixup_criterion(class_logit.double() , target_a , target_b , lam).mean()

            count += 1   
            
        loss = loss_sum / (args.aug_num)
        loss.backward()
        optimizer.step()	
        if args.progress == True:
            loss_meter.add(loss.item())
            tk0.set_postfix(loss=loss_meter.mean)            
        global_step += 1
    return global_step

def validate(eval_loader, model, args, global_step, epoch, num_classes =10):
    meters = AverageMeterSet()    
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            batch_size = targets.size(0)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs,_ = model(inputs)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            meters.update('top1', prec1.item(), batch_size)
            meters.update('error1', 100.0 - prec1.item(), batch_size)
            meters.update('top5', prec5.item(), batch_size)
            meters.update('error5', 100.0 - prec5.item(), batch_size)
    
        print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
              .format(top1=meters['top1'], top5=meters['top5']))


    return meters['top1'].avg, meters['top5'].avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    
    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def extract_features_simp(train_loader,model,args):
    model.eval()
    embeddings_all  = []
    
    with torch.no_grad():    
        for i, (batch_input) in enumerate(train_loader):
            X_n = batch_input[0].to(args.device)
            _ , feats  = model(X_n)   
            embeddings_all.append(feats.data.cpu())         
    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    return embeddings_all

def load_args(args):
    args.workers = 4 * torch.cuda.device_count()
    label_dir = 'data-local/'
    
    if int(args.label_split) < 10:
        args.label_split = args.label_split.zfill(2)

    if args.dataset == "cifar100":
        args.test_batch_size = args.batch_size
        args.labels = '%s/labels/%s/%d_balanced_labels/%s.txt' % (label_dir,args.dataset,args.num_labeled,args.label_split)
  

    elif args.dataset == "cifar10":
        args.test_batch_size = args.batch_size
        args.labels = '%s/labels/%s/%d_balanced_labels/%s.txt' % (label_dir,args.dataset,args.num_labeled,args.label_split)

    elif args.dataset == "miniimagenet":
        args.train_subdir = 'train'
        args.test_batch_size = args.batch_size
        args.labels = '%s/labels/%s/%d_balanced_labels/%s.txt' % (label_dir,args.dataset,args.num_labeled,args.label_split)


    else:
        sys.exit('Undefined dataset!')

    return args



