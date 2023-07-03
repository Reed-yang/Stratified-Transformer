import select
from typing import Tuple

from zmq import device
import torch


def get_target_prototypes(
    feats, prototypes, target, offset, class_num, dist_div=96, ignore_index=-100
):
    assert not torch.isin(target, torch.tensor([5], device=target.device)).any()
    device, dtype = feats.device, feats.dtype
    offset_ = offset.clone()
    offset_[1:] = offset_[1:] - offset_[:-1]
    batch = (
        torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0)
        .long()
        .cuda()
    )
    batch_size = len(offset_)
    num_bt, num_cls, num_proto, num_dim = prototypes.shape
    assert num_bt == batch_size and num_cls == class_num and num_dim == feats.shape[1]

    target_feats = torch.empty(num_bt, num_cls, num_proto, num_dim, device=device)
    target_mean_feats = torch.empty(num_bt, num_cls, num_dim, device=device)
    target_feats_dist = torch.empty(num_bt, num_cls, num_dim, dist_div, device=device)
    mask = torch.zeros(prototypes.shape[:2], device=device).bool()

    for bt in range(batch_size):
        batch_target = target[batch == bt]
        batch_feats = feats[batch == bt]
        unique_labels = torch.unique(batch_target)
        # assert unique_labels.max() < class_num
        unique_labels = unique_labels[
            ~torch.isin(
                unique_labels, torch.tensor([ignore_index, class_num], device=device)
            )
        ]
        for cls in unique_labels:
            cls_idx = torch.nonzero(batch_target == cls).squeeze(1)
            cls_feats = batch_feats[cls_idx]

            mean_feats = torch.mean(cls_feats, dim=0, keepdim=True)
            target_mean_feats[bt][cls] = mean_feats

            select_feats, select_idx = get_target_cls_feats_oneBatch_outstand(
                cls_feats, cls_idx, num_proto
            )
            assert batch_target[select_idx].eq(cls).all()
            target_feats[bt][cls] = select_feats

            # feats_dist = get_target_cls_feats_dist(cls_feats, dist_div)
            # target_feats_dist[bt][cls] = feats_dist

            mask[bt][cls] = True
    # assert target_feats.max() < 100
    # assert target_mean_feats.min() > -100
    return (
        target_feats,
        target_mean_feats,
        # target_feats_dist,
        mask,
    )


def get_target_cls_feats_oneBatch_random(batch_feats, cls_idx, num_proto):
    if len(cls_idx) < num_proto:
        sup_idx = torch.randint(0, len(cls_idx), (num_proto - len(cls_idx),))
        sup_idx = cls_idx[sup_idx]
        cls_idx = torch.cat([cls_idx, sup_idx], dim=0)
    select_idx = cls_idx[torch.randperm(len(cls_idx))[:num_proto]]
    select_feats = batch_feats[select_idx]
    return select_feats, select_idx


def get_target_cls_feats_oneBatch_outstand(cls_feats, cls_idx, num_proto):
    device = cls_feats.device
    num_pts, num_dim = cls_feats.shape

    feats_trans = cls_feats.transpose(0, 1)  # D, Nc
    dim_sort, sort_idx = torch.sort(feats_trans, dim=-1)
    score = (
        torch.arange(num_pts, device=device).flip(0).repeat(num_dim, 1).float()
        / num_pts
    )
    reverse_idx = torch.argsort(sort_idx, dim=-1)
    feats_score = score.gather(dim=1, index=reverse_idx).transpose(0, 1).sum(-1)

    topk_idx = None
    if len(cls_idx) < num_proto:
        ori_idx = torch.arange(len(cls_idx), device=device)
        sup_idx = torch.randint(
            0, len(cls_idx), (num_proto - len(cls_idx),), device=device
        )
        topk_idx = torch.cat([ori_idx, sup_idx], dim=0)
    else:
        topk_idx = torch.cat(
            [
                feats_score.topk(num_proto // 2, largest=True)[1],
                feats_score.topk(num_proto // 2, largest=False)[1],
            ]
        )

    select_feats = cls_feats[topk_idx]
    select_idx = cls_idx[topk_idx]
    return select_feats, select_idx


def get_target_cls_feats_dist(cls_feats, dist_div):
    # cls_feats: [Nc, D]
    device = cls_feats.device
    num_pts, num_dim = cls_feats.shape
    feats_trans = cls_feats.transpose(0, 1).float()  # D, Nc
    feats_sort = torch.sort(feats_trans, dim=1)[0]
    lower = feats_sort[:, 0] - torch.abs(feats_sort[:, 0]) * 0.01
    upper = feats_sort[:, -1] + torch.abs(feats_sort[:, -1]) * 0.01
    feats_div = tensor_linspace(lower, upper, dist_div + 1)
    feats_dist = torch.zeros(num_dim, dist_div, device=device)
    for d in range(num_dim):
        dim_bucket_idx = torch.bucketize(feats_sort[d], feats_div[d])
        bucket, bucket_count = torch.unique(dim_bucket_idx, return_counts=True)
        bucket = bucket - 1
        max_cnt, min_cnt = bucket_count.max(), bucket_count.min()
        bucket_dist = (bucket_count - min_cnt) / (max_cnt - min_cnt + 1)
        feats_dist[d, bucket] = bucket_dist
    return feats_dist


def tensor_linspace(start, end, steps=10):
    """
    reference: https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


if __name__ == "__main__":
    class_num = 30
    num_inst = 128
    feat_dim = 48
    mask_label = [0, 3, 7, 11, 13]
    ignore_label = -100
    num_generate = 40000
    unknown_label = [5]
    prototypes = torch.randn(class_num, num_inst, feat_dim)
    target = torch.randint(0, class_num, (num_generate,))
    target = target[~torch.isin(target, torch.tensor(mask_label))]
    target[torch.isin(target, torch.tensor(unknown_label))] = ignore_label
    feat = torch.randn(num_generate, feat_dim)

    get_target_prototypes(feat, prototypes, target, class_num, ignore_label)
