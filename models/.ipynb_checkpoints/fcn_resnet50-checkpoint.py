import torch
import torch.nn as nn
import torchvision
import os
def create_model(log_dir=None, pretrain=False):
    model = torchvision.models.segmentation.fcn_resnet50(False)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    if pretrain:
        if log_dir==None:
            log_dir = "/root/.cache/torch/hub/checkpoints//fcn_resnet50_coco-1167a1af.pth"
            pth = torch.load(log_dir)
            for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked", "aux_classifier.4.weight", "aux_classifier.4.bias"]:
                del pth[key]
        elif os.path.exists(log_dir):
            pth = torch.load(log_dir)['state_dict']
        else:
            return model
        model.load_state_dict(pth)
    return model