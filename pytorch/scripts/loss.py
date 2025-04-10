import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')      
        # Focal loss focus on hard, misclassified examples during training, which is particularly useful for imbalanced datasets
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss #For easy examples (high pt), this term is small, reducing their contribution to the loss.
        pred_probs = torch.sigmoid(pred)
        dice_loss = 1 - (2 * (pred_probs * target).sum() + 1) / (pred_probs.sum() + target.sum() + 1)  #image segmentation
        return focal_loss.mean() + dice_loss