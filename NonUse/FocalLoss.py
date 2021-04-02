import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
    binary focal loss
    '''
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = torch.Tensor([alpha, 1-alpha])
        self.nllLoss = nn.NLLLoss(weight=self.weight)
        self.gamma = gamma

    def forward(self, input, target):
        softmax = F.softmax(input, dim=1)
        log_logits = torch.log(softmax)
        fix_weights = (1 - softmax) ** self.gamma
        logits = fix_weights * log_logits

        return self.nllLoss(logits, target)