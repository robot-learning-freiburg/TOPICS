import torch.nn as nn
import torch.nn.functional as F
import torch


def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets, mask=None):
        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        labels = targets.clone()  # B, H, W - torch.Size([12, 512, 512])
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        if mask is not None:
            labels[mask] = self.ignore_index
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class MaskCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets, outputs_old=None):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        ### return to normal
        # outputs[:, 0] = inputs[:, 0] - den
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)

        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        labels = targets  # B, H, W
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction='none')
        mask = torch.zeros_like(targets)
        if outputs_old is not None:
            pseudo_label = torch.argmax(outputs_old, dim=1)
            mask[pseudo_label == 0] = 1
            mask[labels > old_cl] = 1
            loss = loss * mask.detach().float()
        if self.reduction == 'mean':
            loss = -torch.mean(loss)
        elif self.reduction == 'sum':
            loss = -torch.sum(loss)
        return loss


class SoftCrossEntropy(nn.Module):

    def __init__(self, masking_value, pseudo_soft, pseudo_soft_factor=1.0):
        super().__init__()
        if pseudo_soft not in ("soft_certain", "soft_uncertain"):
            raise ValueError(f"Invalid pseudo_soft={pseudo_soft}")
        self.pseudo_soft = pseudo_soft
        self.pseudo_soft_factor = pseudo_soft_factor
        self.masking_value = masking_value

    def __call__(self, logits, labels, logits_old, mask_valid_pseudo):
        nb_old_classes = logits_old.shape[1]
        masked_area = (labels < nb_old_classes) | (labels == self.masking_value)
        loss_certain = F.cross_entropy(logits, labels, reduction="none", ignore_index=self.masking_value) # normal CE
        loss_uncertain = - ((torch.softmax(logits_old, dim=1) * torch.log_softmax(logits[:, :nb_old_classes], dim=1)).sum(
            dim=1)) # old-logits CE, don't take new logits into account

        if self.pseudo_soft == "soft_certain":
            mask_certain = ~masked_area
        elif self.pseudo_soft == "soft_uncertain":
            mask_certain = (mask_valid_pseudo & masked_area) | (~masked_area)
            # mask_valid_pseudo = certain old_class predictions

        loss_certain = mask_certain.float() * loss_certain
        loss_uncertain = (~mask_certain).float() * loss_uncertain
        # print(loss_certain, loss_uncertain, self.pseudo_soft_factor)

        return loss_certain + self.pseudo_soft_factor * loss_uncertain