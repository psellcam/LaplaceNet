import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import numpy as np
import time
import faiss
from faiss import normalize_L2
import scipy
import torch.nn.functional as F
import torch
import scipy.stats
import pickle

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, aug_num , e_transform=None, w_transform=None, s_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.aug_num = aug_num
        self.e_transform = e_transform
        self.w_transform = w_transform
        self.s_transform = s_transform

        imfile_name = '%s/images.pkl' % self.root
        if os.path.isfile(imfile_name):
            with open(imfile_name, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.images = None

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
              
        path, target = self.samples[index]   
        
        ### If unlabeled grab pseduo-label
        labeled_case = self.is_labeled[index]  
        if labeled_case == 0:
            target = self.p_labels[index]                   
        
        if self.images is not None:
            sample = Image.fromarray(self.images[index])
        else:
            sample = self.loader(path)          
        
        ### If in feat mode just give base images
        if self.feat_mode == True:
            aug_images = [] 
            e_sample = self.e_transform(sample)
            aug_images.append(e_sample)
            return aug_images
        
        else:                                
            if labeled_case == 1:          
                aug_images = []                        
                for i in range(self.aug_num):
                    t_sample = self.w_transform(sample)
                    aug_images.append(t_sample)                    
            else:
                aug_images = []                        
                for i in range(self.aug_num):
                    t_sample = self.s_transform(sample)
                    aug_images.append(t_sample)
            
            return aug_images , target  
        
    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DBSS(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.pngs

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, labels , load_im , aug_num , e_transform=None, w_transform=None , s_transform=None,
                 loader=default_loader):
        super(DBSS, self).__init__(root, loader, IMG_EXTENSIONS, aug_num,
                                          e_transform = e_transform,
                                          w_transform = w_transform,
                                          s_transform = s_transform)
        self.imgs = self.samples
        self.labeled_idx = []
        self.unlabeled_idx = []
        self.all_labels = []
        self.feat_mode = False
        self.acc = 0
        self.probs_iter = np.zeros((len(self.imgs),len(self.classes)))
        self.p_labels = []
        self.p_weight = np.ones(len(self.imgs))        
        self.label_dis = 1/(len(self.classes)) * np.ones(len(self.classes))
        
        
        self.relabel_dataset(labels)
        if load_im == True:
            self.load_in_memory()

    def relabel_dataset(self,labels):
        
        for idx in range(len(self.imgs)):
            path, orig_label = self.imgs[idx]
            filename = os.path.basename(path)
            self.all_labels.append(orig_label)
            self.p_labels.append(-1)
            if filename in labels:
                self.labeled_idx.append(idx)
                del labels[filename]
            else:
                self.unlabeled_idx.append(idx)
                
        self.is_labeled = np.ones(len(self.imgs),dtype=int)
        unlabeled_idx = np.asarray(self.unlabeled_idx)
        self.is_labeled[unlabeled_idx]= 0
                
    def load_in_memory(self):
        images = []
        for i in range(len(self.imgs)):
            path,target = self.imgs[i]
            path_image = self.loader(path)
            numpy_image = np.array(path_image)
            images.append(numpy_image)
        self.images = images
      
    def one_iter_true(self, X, k = 50, max_iter = 20, l2 = False , index="ip",n_labels=None):
        if l2:
            normalize_L2(X)  
        
        alpha = 0.99
        labels = np.asarray(self.all_labels)
        labeled_idx = np.asarray(self.labeled_idx)        
        unlabeled_idx = np.asarray(self.unlabeled_idx)
        
        # kNN search for the graph
        d = X.shape[1]       
        if index == "ip":
            index = faiss.IndexFlatIP(d)   # build the index
        if index == "l2":            
            index = faiss.IndexFlatL2(d)   # build the index
        ngus=faiss.get_num_gpus()
        index = faiss.index_cpu_to_all_gpus(index,ngpu=ngus)	
        index.add(X) 
        N = X.shape[0]
        D, I = index.search(X, k + 1)
    
        # Create the graph
        D = D[:,1:] 
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))       
        W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)       
       
        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D        

        # Initiliaze the y vector for each class          
        Z = np.zeros((N,len(self.classes)))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn        
        Y = np.zeros((N,len(self.classes)))
        Y[labeled_idx,labels[labeled_idx]] = 1      
        
        for i in range(len(self.classes)):
            f, _ = scipy.sparse.linalg.cg(A, Y[:,i], tol=1e-3, maxiter=max_iter)
            Z[:,i] = f
        Z[Z < 0] = 0 

        ### Extract and test pseduo labels for accuracy
        probs_iter = F.normalize(torch.tensor(Z),1).numpy() 
        probs_iter[labeled_idx] = np.zeros(len(self.classes)) 
        probs_iter[labeled_idx,labels[labeled_idx]] = 1                
        self.probs_iter = probs_iter      
        p_labels = np.argmax(probs_iter,1)
        self.p_labels = p_labels        
        correct_idx = (self.p_labels[unlabeled_idx] == labels[unlabeled_idx])
        self.acc = correct_idx.mean()   
        print("Pseudo Label Accuracy {:.2f}".format(100*self.acc) + "%",end=" ")    
        
        ### Smooth distrabution alignment
        probs_iter_da = self.dis_align(self.probs_iter)
        probs_iter_da[labeled_idx] = np.zeros(len(self.classes)) 
        probs_iter_da[labeled_idx,labels[labeled_idx]] = 1        
        self.probs_iter = probs_iter_da              
        p_labels = np.argmax(probs_iter_da,1)
        self.p_labels = p_labels
        
        correct_idx = (self.p_labels[unlabeled_idx] == labels[unlabeled_idx])
        self.acc = correct_idx.mean()   
        print("With DA {:.2f}".format(100*self.acc) + "%")    
       
        ## entropy weights if needed
        entropy = scipy.stats.entropy(self.probs_iter.T)
        weights = 1 - entropy / np.log(len(self.classes))
        weights[labeled_idx] = 1.0     
        self.p_weight = weights  

    
    
    def dis_align(self,probs_iter):        
        probs_iter = F.normalize(torch.tensor(probs_iter),1).numpy()     
        p_labels = np.argmax(probs_iter,axis=1) 
        labeled_idx = np.asarray(self.labeled_idx)        
        unlabeled_idx = np.asarray(self.unlabeled_idx)
        labels = np.asarray(self.all_labels)
        
        label_dis_l = np.zeros(len(self.classes))    
        for i in labeled_idx:
            label_dis_l[labels[i]] += 1
        label_dis_l = label_dis_l / len(labeled_idx)
        
        for i in range(100):      
            label_dis_u = np.zeros(len(self.classes))            
            for i in unlabeled_idx:
                label_dis_u[p_labels[i]] += 1
            label_dis_u = label_dis_u / len(unlabeled_idx)
                      
            label_dis = np.divide(label_dis_l,label_dis_u+0.0000001)
            label_dis[label_dis > 1.01] = 1.01
            label_dis[label_dis < 0.99] = 0.99
            
            for i in range(len(self.classes)):
                probs_iter[unlabeled_idx,i] = probs_iter[unlabeled_idx,i] * label_dis[i] 
                
            probs_iter = F.normalize(torch.tensor(probs_iter),1).numpy()             
            p_labels = np.argmax(probs_iter,axis=1)            
        return probs_iter