import copy
import mailbox
import numpy as np
from scipy import rand
import torch
import torch.nn.functional as F

# from pykeops.torch import LazyTensor
import time
# from util.tsne_visualization import balanced_downampling, tsne


def feats_sampling_msp(feats, msp, proto, radius):
    ratio = 1 - msp
    direction = torch.randn_like(feats)
    direction = F.normalize(direction, dim=-1)
    feats_sample = feats + (ratio * radius)[:, None] * direction
    return feats_sample

def get_pseudo_feats_old(old_output):
    return old_output.clone()

def get_pseudo_feats_msp(old_output, batch, offset, num_class=20):
    old_output = old_output.clone()
    device = old_output.device
    batch_size = batch.max() + 1
    output_softmax = torch.softmax(old_output, dim=-1)
    output_msp = output_softmax.max(dim=-1)[0]
    output_label = output_softmax.max(dim=-1)[1]
    unique_class = torch.unique(output_label)
    unique_class = unique_class[unique_class != -100]
    cls_mask = (
        torch.zeros_like(output_label, device=device)[None, :]
        .repeat(num_class, 1)
        .bool()
    )
    for uni_cls in unique_class:
        cls_mask[uni_cls] = output_label == uni_cls

    pseudo_feats = torch.zeros_like(old_output, device=device)
    for bt in range(batch_size):
        bt_mask = batch == bt
        bt_label = output_label[bt_mask]
        unique_label = torch.unique(bt_label)
        unique_label = unique_label[unique_label != -100]
        for cls_lb in unique_label:
            bt_cls_mask = bt_mask & cls_mask[cls_lb]
            bt_cls_output = old_output[bt_cls_mask]
            bt_cls_max_out = bt_cls_output.max(dim=-1)[0]
            bt_cls_min_out = bt_cls_output.min(dim=-1)[0]
            bt_cls_radius = bt_cls_max_out - bt_cls_min_out
            bt_cls_msp = output_msp[bt_cls_mask]
            bt_cls_proto = bt_cls_output.mean(dim=0)
            pseudo_feats[bt_cls_mask] = feats_sampling_msp(
                bt_cls_output, bt_cls_msp, bt_cls_proto, bt_cls_radius
            ).to(pseudo_feats)
    return pseudo_feats


def get_pseudo_label_msp(output):
    output = output.clone()
    output_softmax = torch.softmax(output, dim=-1)
    output_msp = output_softmax.max(dim=-1)[0]
    # pseudo_label = output_msp < 0.7
    return output_msp


# def get_pseudo_label_msp(coord, batch, offset, output):
#     import torch_points_kernels as tp
#
#     coord = coord.clone()
#     output = output.clone()
#     batch_size = batch.max() + 1
#     output_softmax = torch.softmax(output, dim=-1)
#     output_msp = output_softmax.max(dim=-1)[0]
#
#     select_idx = []
#     for bt in range(batch_size):
#         bt_coord = coord[batch == bt]
#         bt_output_msp = output_msp[batch == bt]
#         bt_sort_msp, bt_sort_msp_idx = torch.sort(bt_output_msp, dim=-1)
#
#         bt_coord_trans = bt_coord.transpose(0, 1)
#         bt_scene_bound = (bt_coord_trans.min(-1)[0], bt_coord_trans.max(-1)[0])
#         bt_radius = ((bt_scene_bound[1] - bt_scene_bound[0] + 1e-6) / 8).min()
#         rand_min = torch.randint(0, 50, [1])
#         bt_select_idx = [bt_sort_msp_idx[rand_min].item()]
#
#         while len(bt_select_idx) < 4:
#             rand_idx = torch.randint(0, len(bt_sort_msp_idx), [1])
#             rand_msp = bt_sort_msp[rand_idx]
#             if rand_msp > 0.88:
#                 continue
#             rand_coord = bt_coord[rand_idx]
#             select_coord = bt_coord[bt_select_idx]
#             dist = torch.norm((rand_coord - select_coord), 2)
#             if dist.min() > bt_radius / 2:
#                 bt_select_idx.append(rand_idx.item())
#
#         bt_select_idx = torch.tensor(bt_select_idx).to(bt_coord).long()
#         bt_select_coord = bt_coord[bt_select_idx]
#         bt_x = torch.zeros((len(bt_coord),)).to(bt_coord).long()
#         bt_y = torch.zeros((len(bt_select_coord),)).to(bt_select_coord).long()
#         bt_neighbor_idx = tp.ball_query(
#             bt_radius,
#             500,
#             bt_coord,
#             bt_select_coord,
#             mode="partial_dense",
#             batch_x=bt_x,
#             batch_y=bt_y,
#         )[0]
#         bt_neighbor_idx = torch.cat([bt_select_idx[:, None], bt_neighbor_idx], dim=-1)
#         bt_unique_idx = torch.unique(bt_neighbor_idx)
#         bt_unique_idx = bt_unique_idx[bt_unique_idx != -1]
#
#         select_idx.append(bt_unique_idx + sum(offset[:bt]))
#
#     select_idx = torch.cat(select_idx, dim=0)
#     pseudo_label = torch.zeros_like(batch)
#     pseudo_label[select_idx] = 1
#
#     return pseudo_label.bool()


def get_pseudo_label(output, target, class_num, ignore_index=-100):
    output = output.clone()
    target = target.clone()
    device = output.device
    unique_labels = torch.unique(target)
    assert unique_labels.max() < class_num
    unique_labels = unique_labels[unique_labels != ignore_index]
    prototype = torch.zeros((class_num, output.shape[-1]), device=device)
    for label in unique_labels:
        label_mask = target == label
        prototype[label] = output[label_mask].mean(dim=0)
    mask = torch.where(prototype.norm(dim=-1) > 0)[0].repeat(output.shape[0], 1)
    distance = (output.unsqueeze(1) - prototype.unsqueeze(0)).norm(dim=-1)
    # distance = torch.gather(distance, -1, mask.long()).sum(-1)
    min_distance = distance.min(-1)[0]
    max_min_distance = min_distance.max()
    pseudo_label = torch.ones_like(target) * -1
    pseudo_label = torch.where(min_distance > max_min_distance * 0.8, 1, pseudo_label)
    pseudo_label = torch.where(min_distance < max_min_distance * 0.2, 0, pseudo_label)

    return pseudo_label


def get_pseudo_mask(output, target, class_num, epoch, milestones, ignore_index=-100):
    output = output.clone()
    target = target.clone()
    device = output.device
    unique_labels = torch.unique(target)
    assert unique_labels.max() < class_num
    unique_labels = unique_labels[unique_labels != ignore_index]
    prototype = torch.zeros((class_num, output.shape[-1]), device=device)
    for label in unique_labels:
        label_mask = target == label
        prototype[label] = output[label_mask].mean(dim=0)
    mask = torch.where(prototype.norm(dim=-1) > 0)[0]
    distance = (output.unsqueeze(1) - prototype.unsqueeze(0)).norm(dim=-1)
    min_distance = distance[:, mask].min(-1)[0]
    max_min_distance = min_distance.max()
    gamma = 0.8
    if epoch >= milestones[0] and epoch < milestones[1]:
        gamma = 0.98
    elif epoch >= milestones[1]:
        gamma = 0.99
    pseudo_mask = min_distance > max_min_distance * gamma

    return pseudo_mask


def get_pseudo_mask_from_prototypes(
    feats, target, offset, prototypes, class_num, ignore_index=-100
):  # TODO PCA
    device = feats.device
    offset_ = offset.clone()
    offset_[1:] = offset_[1:] - offset_[:-1]
    batch = (
        torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0)
        .long()
        .cuda()
    )
    batch_size = len(offset_)

    pseudo_mask = []
    num_prototypes = prototypes.shape[2]
    for bt in range(batch_size):
        batch_target = target[batch == bt]
        batch_feats = feats[batch == bt].log_softmax(dim=-1)
        batch_prototypes = prototypes[bt].log_softmax(dim=-1)
        batch_pseudo_mask = torch.zeros_like(batch_target).bool()

        down_indices = balanced_downampling(batch_target)
        down_feats = batch_feats[down_indices]
        down_label = batch_target[down_indices]
        tsne(
            down_feats,
            down_label,
            f"/workspace/Open_World/stratified-transformer/saved/outputs/ori.png",
        )

        unique_labels = torch.unique(batch_target)
        unique_labels = unique_labels[unique_labels != ignore_index]
        max_unique_label = unique_labels.max()
        num_unique_label = len(unique_labels)
        assert max_unique_label < class_num

        masked_proto = batch_prototypes[unique_labels].flatten(0, 1)
        masked_proto_labels = unique_labels[:, None].repeat(1, num_prototypes)
        batch_feats = torch.cat([batch_feats, masked_proto], 0)
        assert len(batch_feats) == (batch == bt).sum() + len(masked_proto)
        batch_target[:] = ignore_index
        batch_target = torch.cat([batch_target, masked_proto_labels.flatten()], 0)

        new_down_feats = torch.cat([down_feats, masked_proto], 0)
        new_down_target = torch.cat([down_label, masked_proto_labels.flatten()], 0)
        new_down_target[-len(masked_proto) :] += 100
        tsne(
            new_down_feats,
            new_down_target,
            f"/workspace/Open_World/stratified-transformer/saved/outputs/prototype.png",
        )

        label_remap = 256 * torch.ones(max_unique_label + 1, device=device).long()
        label_remap[unique_labels] = torch.arange(num_unique_label, device=device)
        valid_mask = batch_target != ignore_index
        batch_target[valid_mask] = label_remap[batch_target[valid_mask]]
        assert (batch_target != 256).all() and (batch_target < num_unique_label).all()

        seed_kmeans = ConstrainedSeedKMeans(
            n_clusters=num_unique_label + 1,
            n_init=1,
            max_iter=3,
            verbose=False,
            invalide_label=ignore_index,
        )
        seed_kmeans.fit(batch_feats, batch_target)

        map_label = seed_kmeans.indices
        map_label_ori = map_label[: -len(masked_proto)][down_indices]
        map_label_proto = map_label[-len(masked_proto) :]
        map_label = torch.cat([map_label_ori, map_label_proto], 0)
        unique_labels = torch.cat(
            [unique_labels, torch.tensor([class_num], device=device)]
        )
        map_label = unique_labels[map_label]
        map_label[-len(masked_proto) :] += 100
        tsne(
            new_down_feats,
            map_label,
            f"/workspace/Open_World/stratified-transformer/saved/outputs/kmeans.png",
        )

        group_label = seed_kmeans.indices[: -len(masked_proto)]
        batch_pseudo_mask[group_label == num_unique_label] = True
        pseudo_mask.append(batch_pseudo_mask)

    pseudo_mask = torch.cat(pseudo_mask, dim=0)
    assert pseudo_mask.shape == target.shape
    return pseudo_mask


# def get_pseudo_mask_from_prototypes_oneItem():


class ConstrainedSeedKMeans:
    """Constrained seed KMeans algorithm proposed by Basu et al. in 2002."""

    def __init__(
        self,
        n_clusters=2,
        *,
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=False,
        invalide_label=-1,
    ):
        """Initialization a constrained seed kmeans estimator.

        Args:
            n_clusters: The number of clusters.
            n_init: The number of times the algorithm will run in order to choose
                the best result.
            max_iter: The maximum number of iterations the algorithm will run.
            tol: The convergence threshold of the algorithm. If the norm of a
                matrix, which is the difference between two consective cluster
                centers, is less than this threshold, we think the algorithm converges.
            verbose: Whether to print intermediate results to console.
            invalide_label: Spicial sign to indicate which samples are unlabeled.
                If the y value of a sample equals to this value, then that sample
                is a unlabeled one.
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.INVALID_LABEL = invalide_label

    def _check_params(self, X, y):
        """Check if the parameters of the algorithm and the inputs to it are valid."""
        assert type(X) is torch.Tensor
        assert type(y) is torch.Tensor

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"The number of clusters mube be less than the " f"number of samples."
            )

        if self.max_iter <= 0:
            raise ValueError(f"The number of maximum iteration must larger than zero.")

    def _init_centroids(self, X, y):
        y_unique = torch.unique(y)
        if self.INVALID_LABEL in y_unique:
            n_seed_centroids = len(y_unique) - 1
        else:
            n_seed_centroids = len(y_unique)
        assert n_seed_centroids <= self.n_clusters, (
            f"The number of seed centroids"
            f"should be less than the total"
            f"number of clusters."
        )

        centers = torch.empty(
            (self.n_clusters, X.shape[1]), dtype=X.dtype, device=X.device
        )
        # First, initialize seed centers using samples with label
        for i in range(n_seed_centroids):
            seed_samples = X[y == i]
            centers[i] = seed_samples.mean(axis=0)

        # Then, initilize the remaining centers with random samples from X
        unlabel_idxes = torch.where(y == self.INVALID_LABEL)[0]

        # If all samples are labeled, kmeans algorithm may not updates itself, moreover, there is no need for clustering
        if len(unlabel_idxes) == 0:
            raise ValueError("All samples are labeled! No need for clustering!")

        if len(unlabel_idxes) < self.n_clusters - n_seed_centroids:
            assert True, "unexpected situation"
            # In this case, we randomly select (self.n_clusters - n_seed_centroids) different data points from the whole dataset
            idx = np.random.randint(X.shape[0], size=self.n_clusters - n_seed_centroids)
            idx = torch.randint(
                0, X.shape[0], size=(self.n_clusters - n_seed_centroids,)
            )
            print("index", idx)
            for i in range(n_seed_centroids, self.n_clusters):
                centers[i] = X[idx[i - n_seed_centroids]]
        else:
            for i in range(n_seed_centroids, self.n_clusters):
                # idx = np.random.choice(unlabel_idxes, 1, replace=False)
                idx = torch.randint(0, len(unlabel_idxes), (1,))
                centers[i] = X[idx]

        return centers, n_seed_centroids

    def _kmeans(self, X, y, init_centers):
        """KMeans algorithm implementation."""
        indices = y.clone().detach()
        n_samples, n_features = X.shape[0], X.shape[1]
        cur_centers = init_centers
        new_centers = init_centers.clone().detach()
        # indices_ = indices.clone().detach()
        # Main loop
        for iter_ in range(self.max_iter):
            # Fist step in KMeans: calculate the closest centroid for each sample
            # for i in range(n_samples):
            #     # If this sample has label, then we use the ground-truth label
            #     # as its cluster index
            #     if y[i] != self.INVALID_LABEL:
            #         continue

            #     min_idx = torch.norm(cur_centers - X[i], dim=1).argmin()
            #     indices[i] = min_idx
            mask = y == self.INVALID_LABEL
            indices[mask] = torch.norm(cur_centers - X[mask, None, :], dim=-1).argmin(
                -1
            )
            # assert torch.all(indices == indices_)

            # Second step in KMeans: update each centroids
            for i in range(self.n_clusters):
                cluster_samples = X[indices == i]
                # In the case that the cluster is empty, randomly choose
                # a sample from X.
                if cluster_samples.shape[0] == 0:
                    new_centers[i] = X[np.random.choice(n_samples, 1, replace=False)]
                else:
                    new_centers[i] = cluster_samples.mean(axis=0)

            # Calculate inertial at current iteration
            inertia = 0
            for i in range(self.n_clusters):
                inertia += (
                    torch.norm(X[indices == i] - new_centers[i], dim=1).sum().item()
                )
            if self.verbose:
                print("Iteration {}, inertia: {}".format(iter_, inertia))

            # Check if KMeans converges
            difference = torch.norm(new_centers - cur_centers, p="fro")
            if difference < self.tol:
                if self.verbose:
                    print("Converged at iteration {}.\n".format(iter_))
                break

            # ATTENSION: Avoid using direct assignment like cur_centers = new_centers
            # This will cause cur_centers and new_cneters to point at the same
            # object in the memory. To fix this, you must create a new object.
            cur_centers = new_centers.clone().detach()

        return new_centers, indices, inertia

    def fit(self, X, y):
        """Using features and little labels to do clustering.

        Args:
            X: numpy.ndarray or torch.Tensor with shape (n_samples, n_features)
            y: List or numpy.ndarray, or torch.Tensor with shape (n_samples,).
                For index i, if y[i] equals to self.INVALID_LABEL, then X[i] is
                an unlabels sample.

        Returns:
            self: The estimator itself.
        """
        self._check_params(X, y)

        _, n_seed_centroids = self._init_centroids(X, y)
        if n_seed_centroids == self.n_clusters:
            self.n_init = 1

        # run constrained seed KMeans n_init times in order to choose the best one
        best_inertia = None
        best_centers, best_indices = None, None
        for i in range(self.n_init):
            init_centers, _ = self._init_centroids(X, y)
            if self.verbose:
                print("Initialization complete")
            new_centers, indices, new_inertia = self._kmeans(X, y, init_centers)
            if best_inertia is None or new_inertia < best_inertia:
                best_inertia = new_inertia
                best_centers = new_centers
                best_indices = indices

        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
        self.indices = best_indices

        return self

    def predict(self, X):
        """Predict the associated cluster index of samples.

        Args:
            X: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).

        Returns:
            indices: The associated cluster index of each sample, with shape
            (n_samples,)
        """
        n_samples = X.shape[0]
        indices = [-1 for _ in range(n_samples)]

        for i in range(n_samples):
            if type(X) == np.ndarray:
                min_idx = np.linalg.norm(self.cluster_centers_ - X[i], axis=1).argmin()
            else:
                min_idx = torch.norm(self.cluster_centers_ - X[i], dim=1).argmin()
            indices[i] = min_idx

        if type(X) == np.ndarray:
            return np.array(indices)
        else:
            return torch.tensor(indices)

    def fit_predict(self, X, y):
        """Convenient function."""
        return self.fit(X, y).predict(X)

    def transform(self, X):
        """Transform the input to the centorid space.

        Args:
            X: numpy.ndarray or torch.Tensor with shape (n_samples, n_features).

        Returns:
            output: With shape (n_samples, n_clusters)
        """
        if type(X) == np.ndarray:
            pkg = np
        else:
            pkg = torch

        n_samples = X.shape[0]
        output = pkg.empty((n_samples, self.n_clusters), dtype=X.dtype)
        for i in range(n_samples):
            if type(X) == np.ndarray:
                output[i] = np.linalg.norm(self.cluster_centers_ - X[i], axis=1)
            else:
                output[i] = torch.norm(self.cluster_centers_ - X[i], dim=1)

        return output

    def fit_transform(self, X, y):
        """Convenient function"""
        return self.fit(X, y).transform(X)

    def score(self, X):
        """Opposite of the value of X on the K-means objective."""
        interia = 0
        n_samples = X.shape[0]

        for i in range(n_samples):
            if type(X) == np.ndarray:
                interia += np.linalg.norm(self.cluster_centers_ - X[i], axis=1).min()
            else:
                interia += torch.norm(self.cluster_centers_ - X[i], dim=1).min().item()

        return -1 * interia


if __name__ == "__main__":
    x = torch.randn(10000, 48)
    c_j = torch.randn(20, 48)
