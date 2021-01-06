# import torch
import torch.nn as nn
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1), weights=[0.8,0.2]):

        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, mask):
        probs = logits.sigmoid()
        tp = (probs * mask).sum(self.dims)
        fp = (probs * (1 - mask)).sum(self.dims)
        fn = ((1 - probs) * mask).sum(self.dims)
        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dice_loss = 1- dice.mean()
        bce_loss = self.bce(logits, mask)
        return weights[0]*bce_loss+ dice_loss*weights[1]

def create_loss():
    return DiceBCELoss(smooth=1., dims=(-2,-1))