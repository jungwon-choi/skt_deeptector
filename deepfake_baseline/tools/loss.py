# Copyright © 2020-2021 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch.nn as nn

#===============================================================================
class Weihgted_MSELoss(nn.Module):
    def __init__(self, weight_balance=False, pos_weight=1.):
        super(Weihgted_MSELoss, self).__init__()
        self.weight_balance = weight_balance
        self.pos_weight = pos_weight
        self.criterion = nn.MSELoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

    def forward(self, preds, labels):
        losses = self.criterion(self.sigmoid(preds), labels)
        if self.weight_balance:
            weights = labels.clone()
            weights[weights==0.] = self.pos_weight
            losses = losses*weights
        return losses.mean()
