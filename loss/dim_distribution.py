import torch


def dim_distribution(self, feats, area=96):
        # feats : [N, D]
        num_pts, _ = feats.shape
        feats_dist = torch.zeros(num_pts, self.area, device=feats.device)
        feats_trans = feats.transpose(0, 1) # D, N
        feats_sort = torch.sort(feats_trans,dim=0)
        