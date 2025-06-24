import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    taken from: https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
    '''
    def __init__(self, p=1, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        labels = labels.long()

        num_classes = logits.shape[1]

        if labels.dim() == 3:
            labels_onehot = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
        else:
            raise ValueError(f"Expected labels of shape (N, H, W), got {labels.shape}")

        preds = F.softmax(logits, dim=1)

        numer = (preds * labels_onehot).sum(dim=(0, 2, 3))
        denor = (preds.pow(self.p) + labels_onehot.pow(self.p)).sum(dim=(0, 2, 3))
        dice = (2 * numer + self.smooth) / (denor + self.smooth)

        loss = 1 - dice.mean()
        return loss

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(DiceCrossEntropyLoss, self).__init__()
        self.ce_weight = weight_ce
        self.dice_weight = weight_dice

        self.dice_loss = SoftDiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def forward(self, preds, labels):

        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        labels = labels.long()

        ce_loss = self.ce_loss(preds, labels)
        dice_loss = self.dice_loss(preds, labels)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return loss