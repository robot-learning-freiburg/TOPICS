import torch.nn as nn
import torch

class HierarchicalLoss(nn.Module):
    # adapted from https://github.com/MinaGhadimiAtigh/HyperbolicImageSegmentation/tree/main/hesp
    def __init__(self, hier_matrices, device, ignore_label, no_classes):
        super().__init__()
        self.EPS = 1e-15
        self.device = device
        self.sibling_tree = hier_matrices[0].double().to(device)
        self.hier_tree = hier_matrices[1].double().to(device)
        self.ignore_label = ignore_label
        self.no_classes = no_classes
        self.helper_tree = torch.eye(self.hier_tree.shape[1]).double().to(device)
        for i in hier_matrices[2]:
            if i < self.helper_tree.shape[1]:
                self.helper_tree[i, i] = 0
        if len(hier_matrices) > 3:
            self.sibling_tree_old = hier_matrices[3].double().to(device)
            self.hier_tree_old = hier_matrices[4].double().to(device)
            self.helper_tree_old = torch.eye(self.hier_tree_old.shape[1]).double().to(device)
            for i in hier_matrices[5]:
                if i < self.helper_tree_old.shape[1]:
                    self.helper_tree_old[i, i] = 0

    def __call__(self, logits, labels, old_model=False):
        """ Categorical cross-entropy loss.
        Suppports both flat and hierarchical classification.
        Calculated as -mean(sum(log(p_correct)))

        Args:
            probs: flattened probabilities over H, NxM
            labels: flattened idx of correct class, N
        Returns:
            loss object
        """
        logits = logits.double()
        hier_tree = self.hier_tree_old if old_model else self.hier_tree
        labels_one_hot = self.onehot_with_ignore_label(labels)
        # copied from softmax
        sm_probs = self.softmax(logits, old_model)
        # copied from CCE loss
        log_probs = torch.log(torch.clamp(sm_probs, min=1e-15))
        log_sum_p = torch.tensordot(log_probs, hier_tree, dims=([1], [1]))  # multiply with ancestors
        pos_logp = log_sum_p.masked_select(labels_one_hot.bool())
        loss = -torch.mean(pos_logp)
        return loss

    def softmax(self, logits, old_model=False):
        sibling_tree = self.sibling_tree_old if old_model else self.sibling_tree
        logits = (logits - torch.max(logits, dim=1, keepdim=True)[0]).exp()
        Z = torch.tensordot(logits, sibling_tree, dims=([1], [-1])).permute(0, 3, 1, 2)  # siblings
        log_probs = logits / torch.clamp(Z, min=1e-15)
        assert torch.all(log_probs >= 0)
        assert torch.all(log_probs <= 1)
        return log_probs

    def decide(self, logits, old_model=False):
        logits = logits.double()
        logits = self.softmax(logits, old_model=old_model)
        log_probs = torch.clamp(logits, min=1e-4).log()

        hier_tree = self.hier_tree_old if old_model else self.hier_tree
        helper_tree = self.helper_tree_old if old_model else self.helper_tree

        log_sum_p = torch.tensordot(log_probs, hier_tree, dims=([1], [1])).permute(0, 3, 1, 2).exp()
        cls_probs = torch.tensordot(log_sum_p, helper_tree, dims=([1], [1])).permute(0, 3, 1, 2)

        _, prediction = cls_probs.max(dim=1)
        return cls_probs, prediction

    def onehot_with_ignore_label(self, labels):
        dummy_label = self.no_classes + 1
        mask = labels == self.ignore_label
        modified_labels = labels.clone()
        modified_labels[mask] = self.no_classes
        # One-hot encode the modified labels
        one_hot_labels = torch.nn.functional.one_hot(modified_labels, num_classes=dummy_label)
        # Remove the last row in the one-hot encoding
        one_hot_labels = one_hot_labels[:, :, :, :-1]
        return one_hot_labels.to(self.device)


class HBCELoss(nn.Module):

    def __init__(self, hier_matrices, device, ignore_label, no_classes):
        super().__init__()
        self.EPS = 1e-15
        self.device = device
        self.sibling_tree = hier_matrices[0].double().to(device)
        self.hier_tree = hier_matrices[1].double().to(device)
        self.ignore_label = ignore_label
        self.no_classes = no_classes
        self.bce = nn.BCELoss()
        self.helper_tree = torch.eye(self.hier_tree.shape[1]).double().to(device)
        for i in hier_matrices[2]:
            if i < self.helper_tree.shape[1]:
                self.helper_tree[i, i] = 0
        if len(hier_matrices) > 3:
            self.sibling_tree_old = hier_matrices[3].double().to(device)
            self.hier_tree_old = hier_matrices[4].double().to(device)
            self.helper_tree_old = torch.eye(self.hier_tree_old.shape[1]).double().to(device)
            for i in hier_matrices[5]:
                if i < self.helper_tree_old.shape[1]:
                    self.helper_tree_old[i, i] = 0

    def __call__(self, logits, labels, old_model=False):
        """ Categorical cross-entropy loss.
        Suppports both flat and hierarchical classification.
        Calculated as -mean(sum(log(p_correct)))

        Args:
            probs: flattened probabilities over H, NxM
            labels: flattened idx of correct class, N
        Returns:
            loss object
        """
        labels_one_hot = self.onehot_with_ignore_label(labels)
        # copied from CCE loss
        sig_probs = torch.sigmoid(logits)
        hier_tree = self.hier_tree_old if old_model else self.hier_tree
        labels_one_hot = torch.einsum("inko, op -> ipnk ", labels_one_hot.double(), hier_tree)
        loss = self.bce(sig_probs, labels_one_hot.double())
        loss = loss.masked_select(labels_one_hot.bool())  # remove ignore index
        return loss

    def mix(self, logits, old_model=False):
        hier_tree = self.hier_tree_old if old_model else self.hier_tree
        helper_tree = self.helper_tree_old if old_model else self.helper_tree
        log_sum_p = torch.tensordot(logits, hier_tree, dims=([1], [1])).permute(0, 3, 1, 2)
        cls_probs = torch.tensordot(log_sum_p, helper_tree, dims=([1], [1])).permute(0, 3, 1, 2)
        return cls_probs

    def decide(self, logits, old_model=False):
        logits = self.mix(logits, old_model=old_model)
        prediction = logits.argmax(dim=1)  # pred: [N. H, W]
        return logits, prediction

    def onehot_with_ignore_label(self, labels):
        dummy_label = self.no_classes + 1
        mask = labels == self.ignore_label
        modified_labels = labels.clone()
        modified_labels[mask] = self.no_classes
        # One-hot encode the modified labels
        one_hot_labels = torch.nn.functional.one_hot(modified_labels, num_classes=dummy_label)
        # Remove the last row in the one-hot encoding
        one_hot_labels = one_hot_labels[:, :, :, :-1]
        return one_hot_labels.to(self.device)