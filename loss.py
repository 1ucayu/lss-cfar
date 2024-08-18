import torch
import torch.nn as nn

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        """
        Wrapper for nn.BCEWithLogitsLoss that handles input with batch size in the second dimension.

        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class. If given, has to be a Tensor of size `C`.
            reduction (string, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'
            pos_weight (Tensor, optional): A weight of positive examples. Must be a vector with length equal to the number of classes.
        """
        super(BCEWithLogitsLoss, self).__init__()
        # self.criterion = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction, pos_weight=pos_weight)
        # self.criterion = nn.BCELoss(weight=weight, reduction=reduction)
        # apply L1 loss
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, input, target):
        """
        Computes the loss.

        Args:
            input (Tensor): The input tensor (logits) from the model with shape [H, B, W].
            target (Tensor): The target tensor with shape [H, B, W] with values 0 or 1.

        Returns:
            Tensor: The computed loss.
        """
        # Permute input and target to move batch size to the first dimension
        input = input.permute(1, 0, 2)  # From [H, B, W] to [B, H, W]
        target = target.permute(1, 0, 2)  # From [H, B, W] to [B, H, W]

        return self.criterion(input, target)
    
# Test function
def test():
    criterion = BCEWithLogitsLoss()
    input = torch.randn(87, 4, 142)  # (sequence_length, batch_size, hidden_dimension)
    target = torch.randint(0, 1, (87, 4, 142)).float()  # (sequence_length, batch_size, hidden_dimension)
    loss = criterion(input, target)
    print(loss)

if __name__ == "__main__":
    # test()
    pass

