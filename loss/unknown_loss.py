import torch.nn as nn
import torch


class InfoNCELoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(InfoNCELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, unknown_pred, pseudo_label):
        
        return 0
