import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class CFARCNN(nn.Module):
    def __init__(self):
        super(CFARCNN, self).__init__()
        
        # Convolutional layers with normalization and activation
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for conv1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization for conv2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization for conv3
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)  # Batch normalization for conv4
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)  # Batch normalization for conv5
        
        # Fully connected layer to output binary matrix
        self.fc1 = nn.Conv2d(64, 1, kernel_size=1)  # Output layer

    def forward(self, x):
        # logger.debug(f"Input contains NaNs: {torch.isnan(x).any()}")
        # logger.debug(f"Input contains Infs: {torch.isinf(x).any()}")

        x = self.bn1(self.conv1(x))
        # logger.debug(f"conv1 output min: {x.min()}, max: {x.max()}, NaNs: {torch.isnan(x).any()}, Infs: {torch.isinf(x).any()}")
        x = F.relu(x)
        
        x = self.bn2(self.conv2(x))
        # logger.debug(f"conv2 output min: {x.min()}, max: {x.max()}, NaNs: {torch.isnan(x).any()}, Infs: {torch.isinf(x).any()}")
        x = F.relu(x)
        
        x = self.bn3(self.conv3(x))
        # logger.debug(f"conv3 output min: {x.min()}, max: {x.max()}, NaNs: {torch.isnan(x).any()}, Infs: {torch.isinf(x).any()}")
        x = F.relu(x)
        
        x = self.bn4(self.conv4(x))
        # logger.debug(f"conv4 output min: {x.min()}, max: {x.max()}, NaNs: {torch.isnan(x).any()}, Infs: {torch.isinf(x).any()}")
        x = F.relu(x)
        
        x = self.bn5(self.conv5(x))
        # logger.debug(f"conv5 output min: {x.min()}, max: {x.max()}, NaNs: {torch.isnan(x).any()}, Infs: {torch.isinf(x).any()}")
        x = F.relu(x)
        
        x = torch.sigmoid(self.fc1(x))
        # logger.debug(f"fc1 output min: {x.min()}, max: {x.max()}, NaNs: {torch.isnan(x).any()}, Infs: {torch.isinf(x).any()}")

        return x


# Function to print model parameters
def print_model_parameters(model):
    total_params = 0
    print("Model Parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")
        total_params += param.numel()
    print(f"Total number of parameters: {total_params}")

# Test function with parameter printing
def test_model_with_params():
    # Instantiate the model
    model = CFARCNN()
    # 738K parameters

    # Print model parameters
    print_model_parameters(model)

    # Create a random input tensor of shape [4, 1, 87, 128]
    input_tensor = torch.randn(4, 1, 87, 128)

    # Forward pass through the model
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

if __name__ == "__main__":
    # Run the test with parameters
    test_model_with_params()
