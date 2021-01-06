import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import importlib
import pdb
import math
from bisect import bisect_right
import cv2,numba
import tifffile as tif
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd 

def get_value(oldv, newv):
    if newv is not None:
        return newv
    else:
        return oldv

def update(config, args):
    # Change parameters
    config['model_dir'] = get_value(config['model_dir'], args.model_dir)
    config['training_opt']['batch_size'] = \
        get_value(config['training_opt']['batch_size'], args.batch_size)    
    return config


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def print_write(print_str, log_file):
    print(*print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)
    
    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]    
    return slices.reshape(nx*ny,4)

def read_tiff(image_file):
    image = tiff.imread(image_file)
    if image.shape[0]==3:
        image = image.transpose(1, 2, 0)
        image = np.ascontiguousarray(image)
    return image

def read_json_as_df(json_file):
   with open(json_file) as f:
       j = json.load(f)
   df = pd.json_normalize(j)
   return df

def init_weights(model, weights_path, caffe=False, classifier=False):  
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))    
    weights = torch.load(weights_path)   

    weights = weights['state_dict']['model']
    weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k] 
                for k in model.state_dict()}

    model.load_state_dict(weights)   
    return model


def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p>0.5
    t = t>0.5
    uion = p.sum() + t.sum()
    overlap = (p*t).sum()
    dice = 2*overlap/(uion+0.001)
    return dice.mean()

def np_accuracy(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    p = p>0.5
    t = t>0.5
    tp = (p*t).sum()/t.sum()
    tn = ((1-p)*(1-t)).sum()/(1-t).sum()
    return tp, tn

def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x

# Tang Kaihua New Add
def print_grad_norm(named_parameters, logger_func, log_file, verbose=False):
    if not verbose:
        return None

    total_norm = 0.0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters.items():
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)

    logger_func(['----------Total norm {:.5f}-----------------'.format(total_norm)], log_file)
    for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
        logger_func(["{:<50s}: {:.5f}, ({})".format(name, norm, param_to_shape[name])], log_file)
    logger_func(['-------------------------------'], log_file)

    return total_norm

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_epochs=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def filter_roi(img,tissue_thred=0.9):
    '''如果大于阈值，则说明组织少，过滤该区域
    '''
    w,h,c =img.shape
    bg_R = img[:,:,0]>203
    bg_G = img[:,:,1]>191
    bg_B = img[:,:,2]>201
    tissue_RGB=np.logical_not(bg_B&bg_G&bg_R)
    if tissue_RGB.mean()>tissue_thred:
        return True
    return False

def return_cover(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p>0.5
    t = t>0.5
    uion = p.sum() + t.sum()
    overlap = (p*t).sum()
    return uion,overlap