import torch.nn as nn
def create_loss():
    return nn.BCEWithLogitsLoss()