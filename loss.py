import torch
import torch.nn as nn

# class BCEWithLogitsLoss(nn.Module):
#     def __init__(self, weight=None, reduction='mean', pos_weight=None):
#         """
#         Wrapper for nn.BCEWithLogitsLoss that handles input with batch size in the second dimension.

#         Args:
#             weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
#             reduction (string, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'
#             pos_weight (Tensor, optional): A weight of positive examples. Must be a vector with length equal to the number of classes.
#         """
#         super(BCEWithLogitsLoss, self).__init__()
#         # self.criterion = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction, pos_weight=pos_weight)
#         # self.criterion = nn.BCELoss(weight=weight, reduction=reduction)
#         # apply L1 loss
#         # self.criterion = nn.L1Loss(reduction=reduction)
#         # self.criterion = nn.MSELoss(reduction=reduction)
#         self.criterion = nn.BCELoss(reduction=reduction)

#     def forward(self, input, target):
#         """
#         Computes the loss.

#         Args:
#             input (Tensor): The input tensor (logits) from the model with shape [H, B, W].
#             target (Tensor): The target tensor with shape [H, B, W] with values 0 or 1.

#         Returns:
#             Tensor: The computed loss.
#         """
#         # Permute input and target to move batch size to the first dimension
#         # input = input.permute(1, 0, 2)  # From [H, B, W] to [B, H, W]
#         # target = target.permute(1, 0, 2)  # From [H, B, W] to [B, H, W]

#         return self.criterion(input, target)
    
# Test function
# def test():
#     criterion = BCEWithLogitsLoss()
#     input = torch.randn(87, 4, 142)  # (sequence_length, batch_size, hidden_dimension)
#     target = torch.randint(0, 1, (87, 4, 142)).float()  # (sequence_length, batch_size, hidden_dimension)
#     loss = criterion(input, target)
#     print(loss)

# if __name__ == "__main__":
#     # test()
#     pass

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, loss_type='bce', weight=None, reduction='mean', pos_weight=None, alpha=0.8, gamma=2, smooth=1e-6):
        """
        Custom loss function that supports BCE, Dice, Focal, and BCE + Dice losses.

        Args:
            loss_type (str): Type of loss to use. Options are 'bce', 'dice', 'focal', 'combined'.
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
            pos_weight (Tensor, optional): A weight of positive examples for BCE. Must be a vector with length equal to the number of classes.
            alpha (float, optional): Weighting factor for Focal Loss.
            gamma (float, optional): Focusing parameter for Focal Loss.
            smooth (float, optional): Smoothing factor for Dice Loss.
        """
        super(CustomLoss, self).__init__()
        self.loss_type = loss_type
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction, pos_weight=pos_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)  # Apply sigmoid if logits are used
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)  # Probability of correctly classifying the pixel
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

    def forward(self, input, target):
        if self.loss_type == 'bce':
            return self.bce_loss(input, target)
        elif self.loss_type == 'dice':
            return self.dice_loss(input, target)
        elif self.loss_type == 'focal':
            return self.focal_loss(input, target)
        elif self.loss_type == 'combined':
            bce_loss = self.bce_loss(input, target)
            dice_loss = self.dice_loss(input, target)
            return bce_loss + dice_loss
        elif self.loss_type == 'l1':
            return F.l1_loss(input, target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Choose from 'bce', 'dice', 'focal', 'combined'.")

# Usage example:
# loss_fn = CustomLoss(loss_type='combined')


