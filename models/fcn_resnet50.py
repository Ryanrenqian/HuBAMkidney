import torch
import torch.nn as nn
import torchvision
import os
from utils import init_weights
def create_model(test=False, log_dir=None, pretrain=False):
    model = torchvision.models.segmentation.fcn_resnet50(False)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    if test:
        if not log_dir.endswith('pth'):
            log_dir = log_dir +'/final_model_checkpoint.pth'
        if os.path.exists(log_dir):
            model = init_weights(model,log_dir)
    return model