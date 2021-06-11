import argparse
from . import  datasets

def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: imagenet)')
    parser.add_argument('--model', type=str, help='the model architecture to use')
    parser.add_argument('--train-subdir', type=str, default='train+val',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='test',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--label-split', default=10, type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=16, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled instances')  
    parser.add_argument('--num-steps', type=int, default=250000,
                        help='number of optimisation steps') 
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='mixup alpha for beta dis') 
    parser.add_argument('--aug-num', default=1, type=int , help="number of augs")
    parser.add_argument('--knn', default=50, type=float,
                        metavar='Neighest Neighbourhours', help='graph k-nn')
    parser.add_argument('--progress', default=False, type=bool, help='progress bar on or off')
    return parser


def parse_commandline_args():
    return create_parser().parse_args()

