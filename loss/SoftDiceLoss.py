# import torch
import torch.nn as nn
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    
    def forward(self, logits, mask):
        probs = logits.sigmoid()
        tp = (probs * mask).sum(self.dims)
        fp = (probs * (1 - mask)).sum(self.dims)
        fn = ((1 - probs) * mask).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

def create_loss():
    return SoftDiceLoss(smooth=1., dims=(-2,-1))