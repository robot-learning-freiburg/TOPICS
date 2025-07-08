import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from .dice import DiceCrossEntropyLoss

class HIERALoss(nn.Module):

    def __init__(self, hier_matrices, device, ignore_label, no_classes, adj_dict, 
                 factor=5.0, hiera_feat=False, dice=0.0):
        """ 
        hier_matrices[0]: binary matrix with all sibling classes marked as 1 (n_classes x n_classes)
        hier_matrices[1]: binary matrix with all ancestor classes marked as 1 (n_classes x n_classes)
        hier_matrices[2]: list of non-leaf nodes (e.g. [1,2,3,4])
        
        """
        super().__init__()
        self.EPS = 1e-15
        self.device = device
        self.sibling_tree = hier_matrices[0].double().to(device) # binary matrix with all sibling classes marked with ones
        self.hier_tree = hier_matrices[1].double().to(device) # binary matrix with all ancestor classes and itself marked with 1 
        self.ignore_label = ignore_label
        self.no_classes = no_classes
        self.helper_tree_child = torch.eye(self.hier_tree.shape[1]).double().to(device)
        for i in hier_matrices[2]: # list of all non-leaf nodes (e.g. [1, 2, 3, 4, 5])
            # mark non-leave nodes as child
            if i < self.helper_tree_child.shape[1]:
                self.helper_tree_child[i, i] = 0

        if len(hier_matrices) > 3:
            self.sibling_tree_old = hier_matrices[3].double().to(device)
            self.hier_tree_old = hier_matrices[4].double().to(device)
            self.helper_tree_child_old = torch.eye(self.hier_tree_old.shape[1]).double().to(device)
            for i in hier_matrices[5]:
                if i < self.helper_tree_child_old.shape[1]:
                    self.helper_tree_child_old[i, i] = 0

        self.eps = 1e-8
        if dice > 0:
             self.ce = DiceCrossEntropyLoss(weight_ce=1-dice, weight_dice=dice)
             print("Balanced Dice Loss activated!", dice)
             self.dice = True
        else:
            self.ce = nn.CrossEntropyLoss(reduction='mean')
            self.dice = False
        self.adj_dict = adj_dict
        self.step = 0
        self.factor = factor
        self.hiera_feat = hiera_feat

    def update_param(self, epoch, iter, max_iter):
        self.step = (epoch * max_iter) + iter

    def __call__(self, logits, labels, feats=None):
        """ Categorical cross-entropy loss.
        Suppports both flat and hierarchical classification.
        Calculated as -mean(sum(log(p_correct)))

        Args:
            probs: flattened probabilities over H, NxM
            labels: flattened idx of correct class, N
        Returns:
            loss object
        """
        void_indices = (labels == self.ignore_label)
        b, _, h, w = logits.shape
        logits = torch.sigmoid(logits.float())
        labels_one_hot = self.onehot_with_ignore_label(labels)
        labels_one_hot_all_lvl = torch.einsum("ionk, op -> ipnk ", labels_one_hot.double(), self.hier_tree)

        MCMA, MCLA, cls, labels_hier = [], [], [], []
        for lvl in range(len(self.adj_dict) - 1, 0, -1):
            adj_lvl = self.adj_dict[lvl]
            if len(adj_lvl) == 0:
                    continue
            child_classes = [x for xs in list(adj_lvl.values()) for x in xs]
            MCMA.append(logits[:, child_classes, :, :])
            MCLA.append(logits[:, child_classes, :, :])
            labels_hier.append(labels_one_hot_all_lvl[:, child_classes])
            cls.append(len(child_classes))

            lvl_id = len(cls) - 1
            for index, value in enumerate(child_classes):
                # MCMA: max of myself or any child!
                child_indices = torch.nonzero(self.hier_tree[:, value])[1:, 0]
                if len(child_indices) == 0:
                    MCMA[lvl_id][:, index] = logits[:, value]
                else:
                    child_logits = logits[:, child_indices]
                    parent_logits = logits[:, value].unsqueeze(1)
                    MCMA[lvl_id][:, index] = torch.max(torch.cat([parent_logits, child_logits], dim=1), 1, True)[0][:, 0]
                
                # MCLA: min of myself or any ancestor!
                ancestor_id = torch.nonzero(self.hier_tree[value])[:-1]
                if len(ancestor_id) == 0:
                        continue
                ancestor_logits = logits[:, ancestor_id, :, :][:, 0]
                MCLA[lvl_id][:, index] = \
                    torch.min(torch.cat([ancestor_logits, MCLA[lvl_id][:, index].unsqueeze(1)], dim=1), 1, True)[0][:,0]

        valid_indices = (~void_indices).unsqueeze(1)
        num_valid = valid_indices.sum()

        loss = torch.tensor(0).to(self.device).double()
        ce_loss = torch.tensor(0).to(self.device).double()
        for i in range(len(cls)):
            if self.factor > 0:
                loss += ((-labels_hier[i] * torch.log(MCLA[i] + self.eps)
                                - (1.0 - labels_hier[i]) * torch.log(1.0 - MCMA[i] + self.eps))
                                * valid_indices).sum() / num_valid / cls[i]
            targets = torch.argmax(labels_hier[i], dim=1) if self.dice else labels_hier[i].double()
            ce_loss += self.ce(MCMA[i], targets)

        if self.factor == 0:
                loss = ce_loss
        else:
                loss = self.factor * loss + ce_loss
        if (feats is not None) and self.hiera_feat:
                factor = 1 / 4 * (1 + torch.cos(
                    torch.tensor((self.step - 80000) / 80000 * math.pi))) if self.step < 80000 else 0.5
                loss += (factor * self.tree_triplet_loss(feats, labels))
        return loss

    def tree_triplet_loss(self, feats, labels, max_triplet=200):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        labels = labels.view(-1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(-1, feats.shape[-1])

        triplet_loss = 0
        exist_classes = torch.unique(labels)
        exist_classes = [x for x in exist_classes if x != 255]
        if len(exist_classes) < 3:
            return triplet_loss
        class_count = 0

        for ii in exist_classes:
            index_anchor = labels == ii
            lvl = next((item for i, item in enumerate(self.adj_dict) if
                        ii in [x for xs in list(item.values()) for x in xs]), None)
            if lvl is None:
                print(ii)
                continue
            siblings = [i for i in lvl.values() if ii in i]
            if len(siblings) == 0:
                continue
            siblings = siblings[0]
            index_pos = (labels == any(siblings)) & ~index_anchor
            index_neg = ~index_pos  # all other

            min_size = min(torch.sum(index_anchor), torch.sum(index_pos), torch.sum(index_neg), max_triplet)

            if min_size > 0:
                feats_anchor = feats[index_anchor][:min_size]
                feats_pos = feats[index_pos][:min_size]
                feats_neg = feats[index_neg][:min_size]

                distance = torch.zeros(min_size, 2).cuda()
                distance[:, 0:1] = 1 - (feats_anchor * feats_pos).sum(1, True)
                distance[:, 1:2] = 1 - (feats_anchor * feats_neg).sum(1, True)

                # margin always 0.1 + (4-2)/4 since the hierarchy is three level
                margin = 0.6 * torch.ones(min_size).cuda()

                tl = distance[:, 0] - distance[:, 1] + margin
                tl = F.relu(tl)

                if tl.size(0) > 0:
                    triplet_loss += tl.mean()
                    class_count += 1
        if class_count == 0:
            return triplet_loss
        triplet_loss /= class_count
        return triplet_loss

    def onehot_with_ignore_label(self, labels):
        dummy_label = self.no_classes + 1
        mask = labels == self.ignore_label
        modified_labels = labels.clone()
        modified_labels[mask] = self.no_classes
        # One-hot encode the modified labels
        one_hot_labels = torch.nn.functional.one_hot(modified_labels, num_classes=dummy_label)
        # Remove the last row in the one-hot encoding
        one_hot_labels = one_hot_labels[:, :, :, :-1].permute(0, 3, 1, 2)
        return one_hot_labels.to(self.device)

    def decide(self, logits, old_model=False, keep_parent_logit=False):
        logits = logits.double()
        helper_tree = self.helper_tree_child_old if old_model else self.helper_tree_child
        hier_tree = self.hier_tree_old if old_model else self.hier_tree
        logits = torch.tensordot(logits, hier_tree, dims=([1], [1])).permute(0, 3, 1, 2)
        if not keep_parent_logit:
            logits = torch.tensordot(logits, helper_tree, dims=([1], [1])).permute(0, 3, 1, 2)
        prediction = logits.argmax(dim=1)  # pred: [N. H, W]
        return logits, prediction