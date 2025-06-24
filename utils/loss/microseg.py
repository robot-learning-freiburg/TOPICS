import torch
import torch.nn as nn
from torch.nn import functional as F


class UnseenAugLoss(nn.Module):
    def __init__(self):
        super(UnseenAugLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()

    def forward(self, predict):
        """
        :param predict: a unknown mask prediction b n h w
        :return: loss
        """

        x = predict.view(predict.shape[0], predict.shape[1], -1)  # b n hw
        x = F.normalize(x, p=2, dim=2)
        simmat = torch.einsum('bic,bjc->bij', x, x.detach())  # b n n
        logits = -torch.log(torch.softmax(simmat, dim=-1))
        l = [torch.diag(i) for i in logits]
        l = torch.stack(l)
        loss = torch.mean(l)
        return loss
