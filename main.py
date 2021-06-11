import os
import time
import numpy as np
import torch
from config import datasets, cli
import helpers
import math

def laplacenet():
	#### Get the command line arguments
	args = cli.parse_commandline_args()
	args = helpers.load_args(args)
	args.file = args.model + "_" + args.dataset + "_" + str(args.num_labeled) + "_" + str(args.label_split) + "_" + str(args.num_steps) + "_" + str(args.aug_num) +  ".txt"
	    
	#### Load the dataset
	dataset_config = datasets.__dict__[args.dataset]()
	num_classes = dataset_config.pop('num_classes')
	args.num_classes = num_classes

	#### Create loaders
	#### train_loader loads the labeled data , eval loader is for evaluation
	#### train_loader_noshuff extracts features 
	#### train_loader_l, train_loader_u together create composite batches
	#### dataset is the custom dataset class
	train_loader, eval_loader , train_loader_noshuff , train_loader_l , train_loader_u , dataset = helpers.create_data_loaders_simple(**dataset_config, args=args)

	#### Create Model and Optimiser 
	args.device = torch.device('cuda')
	model = helpers.create_model(num_classes,args)
	optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9,weight_decay=args.weight_decay, nesterov=args.nesterov)

	#### Transform steps into epochs
	num_steps = args.num_steps
	ini_steps = math.floor(args.num_labeled/args.batch_size)*100
	ssl_steps = math.floor( len(dataset.unlabeled_idx) / ( args.batch_size - args.labeled_batch_size))
	args.epochs = 10 + math.floor((num_steps - ini_steps) / ssl_steps)
	args.lr_rampdown_epochs = args.epochs + 10


	#### Information store in epoch results and then saved to file
	global_step = 0
	epoch_results = np.zeros((args.epochs,6))

	#%%
	for epoch in range(args.epochs):
            start_epoch_time = time.time()
	    #### Extract features and run label prop on graph laplacian
            if epoch >= 10:
                dataset.feat_mode = True
                feats = helpers.extract_features_simp(train_loader_noshuff,model,args)  
                dataset.feat_mode = False          
                dataset.one_iter_true(feats,k = args.knn, max_iter = 30, l2 = True , index="ip") 
		
	    #### Supervised Initilisation vs Semi-supervised main loop
            start_train_time = time.time()  
            if epoch < 10:
                print("Supervised Initilisation:", (epoch+1), "/" , 10 )
                for i in range(10):
                    global_step = helpers.train_sup(train_loader, model, optimizer, epoch, global_step, args)                     
                if epoch >= 10:
                    global_step = helpers.train_semi(train_loader_l, train_loader_u, model, optimizer, epoch, global_step, args)  

            end_train_time = time.time()
            print("Evaluating the primary model:", end=" ")
            prec1, prec5 = helpers.validate(eval_loader, model, args, global_step, epoch + 1, num_classes = args.num_classes)

            epoch_results[epoch,0] = epoch
            epoch_results[epoch,1] = prec1
            epoch_results[epoch,2] = prec5 
            epoch_results[epoch,3] = dataset.acc
            epoch_results[epoch,4] = time.time() - start_epoch_time
            epoch_results[epoch,5] = end_train_time - start_train_time
		
            np.savetxt(args.file,epoch_results)



if __name__ == '__main__':
    laplacenet()



