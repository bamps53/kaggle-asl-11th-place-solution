import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LabelSmoothBCEWithLogitsLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        smooth_labels = torch.where(labels == 1, 1 - self.epsilon, self.epsilon / (labels.shape[1] - 1))
        loss = F.binary_cross_entropy_with_logits(logits, smooth_labels)
        return loss
