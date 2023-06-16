import torch
import torch.nn as nn
from chamfer_distance import ChamferDistance as chamfer_dist


class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()
        self.chd = chamfer_dist()

    def forward(self, pred, target):
        """
        pred: (B, N, 3)
        target: (B, M, 3)
        """
        dist1, dist2, idx1, idx2 = self.chd(pred.float(), target.float())
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        # assert loss < 100  # TODO remove
        return loss
