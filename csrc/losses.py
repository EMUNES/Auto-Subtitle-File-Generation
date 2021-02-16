# Some codes FROM: https://github.com/koukyo1994/kaggle-birdcall-6th-place/blob/master/src/criterion.py
# THANKS a lot!

"""
Loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, input, target):
        input_ = input["clipwise_output"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()

        input_ = torch.clamp(input_, 0.0, 1.0)
        
        return self.bce(input_, target)
    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
            ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


class ImprovedFocalLoss(nn.Module):
    def __init__(self, weights=[1, 1]):
        super().__init__()

        self.focal = FocalLoss()
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        normal_loss = self.focal(input_, target)
        auxiliary_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * normal_loss + self.weights[1] * auxiliary_loss


class ImprovedPANNsLoss(nn.Module):
    def __init__(self, output_key="logit", weights=[1, 1]):
        super().__init__()

        self.output_key = output_key
        if output_key == "logit":
            self.normal_loss = nn.BCEWithLogitsLoss()
        else:
            self.normal_loss = nn.BCELoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input[self.output_key]
        target = target.float()

        framewise_output = input["framewise_output"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        normal_loss = self.normal_loss(input_, target)
        # auxiliary_loss = self.bce(clipwise_output_with_max, target)
        auxiliary_loss = self.normal_loss(clipwise_output_with_max, target)

        return self.weights[0] * normal_loss + self.weights[1] * auxiliary_loss
