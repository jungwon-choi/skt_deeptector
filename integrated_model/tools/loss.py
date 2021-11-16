# Copyright © 2020-2021 Jungwon Choi <jungwon.choi@kaist.ac.kr>
import torch
import torch.nn as nn
import torch.nn.functional as F

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

#===============================================================================
def contrastive_loss(features, args):

    # feature : [2*batch, dim]
    #batch_double = features.shape[0]
    #batch = int(batch_double/2)
    #features_key = features[batch:] # [batch, dim]
    #features = F.normalize(features, dim=1)
    #similarity_matrix = torch.matmul(features, features.T) # [batch, 2*batch]

    # discard the main diagonal from both: labels and similarities matrix
    #mask = torch.eye(batch_double).bool().to(args.device)

    #logits = similarity_matrix[~mask].view(batch_double, -1)  # [2*batch, 2*batch-1]
    #logits = logits / args.temperature
    
    #label_1 = torch.arange(batch)+batch-1
    #label_2 = torch.arange(batch)
    #label = torch.cat((label_1, label_2), dim=0).to(args.device)

    labels = torch.cat([torch.arange(args.batch_size) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature
    return logits, labels

