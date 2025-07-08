import torch.nn as nn
import torch
import numpy as np


class DistanceSim(nn.Module):
    def __init__(self, old_classes, internal_masking_value=0, masking_value=0, all_sim=None):
        super(DistanceSim, self).__init__()
        self.old_classes = old_classes
        self.internal_masking_value = internal_masking_value
        self.masking_value = masking_value
        self.mse = nn.MSELoss(reduction='none')
        self.all_sim = all_sim
        print(f"All_sim activated {all_sim}!")

    def forward(self, features, labels, features_old, manifold):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (features.shape[2], features.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == features.shape[-1], '{} {}'.format(labels.shape, features.shape)

        mask = (labels < self.old_classes) & (labels != self.internal_masking_value) & (labels != self.masking_value)
        if self.all_sim == "dist0":
            dist0_old = manifold.dist0(features_old, dim=1)
            dist0_new = manifold.dist0(features, dim=1)
            loss = self.mse(dist0_old, dist0_new)
            dist_loss = loss * mask
        else:
            print("similarity type unknown!")
            exit()
        assert torch.isfinite(dist_loss).all()
        return dist_loss.mean()


class TripletLoss(nn.Module):
    # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/BatchAllTripletLoss.py#L104

    def __init__(self, valid_indices, device, top=5, hier_matrices=None, trip_increment_only=False,
                 abs=False, remove_hier=False, hier_trip=False, no_margin=False, margin_factor=1.0, norm=False,
                 trip_sel='all', tau=0.07):
        super(TripletLoss, self).__init__()
        self.valid_indices = valid_indices
        self.margin = 0.
        self.device = device
        self.top = top
        self.hier_trip = hier_trip
        self.no_margin = no_margin
        self.margin_factor = margin_factor
        if hier_trip:
            self.class_hier = hier_matrices.sum(dim=1)
        self.trip_increment_only = trip_increment_only
        self.abs = abs
        self.norm = norm
        self.trip_sel = trip_sel
        self.min_dist = 0.0
        self.tau = tau
        if abs:
            self.min_dist = 1.0
        self.remove_hier = remove_hier
        if remove_hier:
            self.hier_matrices = hier_matrices.bool().to(device)
        print(f"Top {self.top}!")

    def update_distance(self, offset, normal, manifold, logger=None):
        if self.trip_increment_only:
            no_base = len(offset[0])
        offset = torch.cat(list(offset.parameters()))
        normal = torch.cat(list(normal.parameters()))
        # distances offset to planes
        dist_matrix_old, valid_dist = self.compute_rel(offset, normal, manifold)
        if not self.no_margin:
            self.margin = dist_matrix_old[valid_dist].std() * self.margin_factor
        # normalize distances (0-1)
        if self.norm:
            dist_matrix_old_rel = dist_matrix_old
        else:
            dist_matrix_old_rel = torch.nn.functional.normalize(dist_matrix_old, p=1, dim=1)
        if valid_dist is not None:
            dist_matrix_old_rel[~valid_dist] = self.min_dist
        # remove bkg (don't contrain background as closest as it changes)
        dist_matrix_old_rel[:, 0] = torch.ones(dist_matrix_old.shape[1]) * self.min_dist
        if (not self.abs) and (not self.hier_trip):
            dist_matrix_old_rel[:, :3] = torch.ones(dist_matrix_old.shape[1], 3) * self.min_dist
        if self.remove_hier:  # remove hyperplanes with same ancestors from positives
            dist_matrix_old_rel[self.hier_matrices] = self.min_dist
        # find topk distances
        if self.abs:
            labels = dist_matrix_old_rel.topk(dist_matrix_old.shape[1], dim=1)[1][:, -int(self.top):]  # smallest x
        else:
            labels = dist_matrix_old_rel.topk(dist_matrix_old.shape[1], dim=1)[1][:, :int(self.top)]  # top x largest
        # convert labels to indices for comparison
        self.valid_indices = self.get_triplet_mask(labels, mask=~(dist_matrix_old_rel == self.min_dist))
        if self.trip_increment_only:
            self.valid_indices[:no_base, :, :] = 0
        # short CHECK for correctness, triplet_loss should be 0!
        dist_matrix, _ = self.compute_rel(offset, normal, manifold)
        # shape: (batch_size, batch_size, 1)
        anchor_positive_dists = dist_matrix.unsqueeze(2)
        # # shape: (batch_size, 1, batch_size)
        anchor_negative_dists = dist_matrix.unsqueeze(1)
        # # get loss values for all possible n^3 triplets
        # # shape: (batch_size, batch_size, batch_size)
        if self.abs:
            triplet_loss = anchor_positive_dists - anchor_negative_dists
        else:
            triplet_loss = - anchor_positive_dists + anchor_negative_dists
        triplet_loss = self.valid_indices.float() * triplet_loss
        triplet_loss[triplet_loss < 0] = 0
        assert triplet_loss.sum() < 1e-16
        if logger is not None:
            logger.info(self.valid_indices)

    def get_triplet_mask(self, labels, mask=None):
        """compute a mask for valid triplets
            Args:
              labels: Batch of integer labels. shape: (batch_size,)
            Returns:
              Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
              A triplet is valid if:
              `labels[i] == labels[j] and labels[i] != labels[k]`
              and `i`, `j`, `k` are different.
            """
        # [anchor, negatives, positives]

        # step 1 - get a mask for distinct indices (i != j and j != k)
        if mask is not None:  # remove hyperplanes from which positives are removed.
            indices_not_equal = mask
        else:
            indices_not_equal = ~torch.eye(labels.size(0), device=labels.device).bool()

        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)
        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        # convert labels to binary mask
        label_equal = torch.zeros(labels.size(0), labels.size(0), device=labels.device).bool()
        label_equal[[i for i in range(labels.size(0)) for _ in range(labels.size(1))], labels.flatten().tolist()] = 1
        i_equal_j = label_equal.unsqueeze(2)  # positives
        i_equal_k = label_equal.unsqueeze(1)  # negatives
        valid_labels = ~i_equal_k & i_equal_j
        valid_labels *= distinct_indices
        # remove void from anchor, bkg is not restricted
        valid_labels[0, :, :] = 0
        return valid_labels

    def compute_rel(self, offset, normal, manifold):
        dist_matrix = torch.zeros((offset.shape[0], offset.shape[0])).to(self.device)
        conformal_factor = 1 - manifold.c * offset.pow(2).sum(dim=1, keepdim=True)
        normal = (normal * conformal_factor).float()
        offset = offset.float()
        for i, off in enumerate(offset):
            if self.hier_trip:
                lvl = self.class_hier[i].item()
                sel_normal = normal[self.class_hier == lvl]
                sel_offset = offset[self.class_hier == lvl]
                dist_matrix[i, self.class_hier == lvl] = manifold.dist2plane(x=off, a=sel_normal, p=sel_offset,
                                                                             signed=True, scaled=False)
            else:
                sel_normal = normal
                sel_offset = offset
                dist_matrix[i] = manifold.dist2plane(x=off, a=sel_normal, p=sel_offset, signed=True, scaled=False)
        # exponential (dist 0 = 1, all positive values)!
        valid_dist = ~(dist_matrix == 0)
        if self.abs:
            dist_matrix = dist_matrix.abs()
        elif self.norm:
            dist_matrix -= dist_matrix.min(1, keepdim=True)[0]
            dist_matrix /= dist_matrix.max(1, keepdim=True)[0]
        else:
            dist_matrix = dist_matrix.exp()
        return dist_matrix, valid_dist

    def forward(self, offset, normal, curv):
        offset = torch.cat(list(offset[:-1].parameters()))
        normal = torch.cat(list(normal[:-1].parameters()))
        dist_matrix, _ = self.compute_rel(offset, normal, curv)
        if self.trip_sel == 'infonce':
            exp_dist = ((1 - (dist_matrix / dist_matrix.max(1, keepdim=True)[0])) / self.tau).exp()
            a, p = torch.where(self.valid_indices.sum(dim=2))
            infonce = exp_dist[a, p]
            for i in range(a.shape[0]):
                negs = torch.where(self.valid_indices[a[i], p[i]])[0]
                infonce_denom = infonce[i] + exp_dist[a[i], negs].sum()
                infonce[i] /= infonce_denom
            infonce = -infonce.log()
            # num_positive_triplets = infonce.nonzero().shape[0]
            triplet_loss = infonce.mean()
            if self.tau == 0.5:
                triplet_loss = triplet_loss / 100.
        elif self.trip_sel == 'infonce2':
            exp_dist = (-(dist_matrix) / self.tau)
            max_val = exp_dist.max()
            exp_dist = (exp_dist - max_val).exp()
            a, p = torch.where(self.valid_indices.sum(dim=2))
            infonce = exp_dist[a, p]
            for i in range(a.shape[0]):
                negs = torch.where(self.valid_indices[a[i], p[i]])[0]
                infonce_denom = infonce[i] + exp_dist[a[i], negs].sum()
                infonce[i] /= infonce_denom
            infonce = -infonce.log()
            num_positive_triplets = infonce.nonzero().shape[0]
            triplet_loss = infonce.mean() / 10.
            if self.tau == 0.5:
                triplet_loss = triplet_loss / 100.
        else:
            # shape: (batch_size, batch_size, 1)
            # print(dist_matrix.max())
            anchor_positive_dists = dist_matrix.unsqueeze(2)
            # # shape: (batch_size, 1, batch_size)
            anchor_negative_dists = dist_matrix.unsqueeze(1)
            # # get loss values for all possible n^3 triplets
            # # shape: (batch_size, batch_size, batch_size)
            if self.abs:
                triplet_loss = self.margin + anchor_positive_dists - anchor_negative_dists
            else:
                triplet_loss = self.margin - anchor_positive_dists + anchor_negative_dists
            triplet_loss = self.valid_indices.float() * triplet_loss
            triplet_loss[triplet_loss < 0] = 0

            valid_triplets = triplet_loss[triplet_loss > 1e-16]
            num_positive_triplets = valid_triplets.size(0)

            if self.no_margin:
                triplet_loss = triplet_loss.sum() * 100.

            if self.trip_sel == 'all':
                triplet_loss = valid_triplets.mean()
            # Hardest triplets
            elif self.trip_sel == 'hard':
                triplet_loss = triplet_loss.amax((1, 2)).mean()
            elif self.trip_sel == 'random':
                a, p = torch.where(triplet_loss.sum(dim=2))
                selected_negs = []
                for i in range(a.shape[0]):
                    possible_negs = torch.where(triplet_loss[a[i], p[i]])[0]
                    random_neg = np.random.choice(possible_negs.cpu().numpy())
                    selected_negs.append(random_neg)
                n = torch.tensor(selected_negs, dtype=a.dtype, device=a.device)
                triplet_loss = triplet_loss[a, p, n].mean()
        if triplet_loss > 0:
            return triplet_loss
        else:
            return None
