import torch
import torch.nn as nn
import torch.nn.functional as F

class CFARRNN(nn.Module):
    def __init__(self, cnn_output_size=128, rnn_hidden_size=64, rnn_num_layers=2, input_size=(87, 128)):
        super(CFARRNN, self).__init__()
        
        # CNN layers with further reduced filters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Reduced filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced filters
        self.conv3 = nn.Conv2d(32, cnn_output_size, kernel_size=3, padding=1)  # Reduced filters
        self.pool = nn.MaxPool2d(2, 2)
        
        # Compute the size after the CNN
        self.cnn_output_dim = (input_size[0] // 2) * (input_size[1] // 2) * cnn_output_size
        
        # RNN layers with further reduced hidden size
        self.rnn = nn.LSTM(cnn_output_size, rnn_hidden_size, rnn_num_layers, batch_first=True, bidirectional=False)
        
        # Fully connected layers with reduced size
        self.fc1 = nn.Linear(rnn_hidden_size  * self.cnn_output_dim // cnn_output_size, rnn_hidden_size // 2)  # Reduced size
        self.fc2 = nn.Linear(rnn_hidden_size // 2, input_size[0] * input_size[1])

    def forward(self, x):
        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # Flatten and transpose to match RNN input shape
        x = x.view(x.size(0), x.size(1), -1)  # [b, c, h*w]
        x = x.transpose(1, 2)  # [b, h*w, c]
        
        # Apply RNN layers
        x, _ = self.rnn(x)  # [b, h*w, hidden_size * 2]
        
        # Flatten the RNN output to match the input of the FC layer
        x = x.contiguous().view(x.size(0), -1)
        
        # Apply Fully Connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Reshape output to original input size
        x = x.view(x.size(0), 1, 87, 128)  # Assuming original input size [87, 128]
        x = torch.sigmoid(x)  # Binary classification (0 or 1)
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
    model = CFARRNN()

    # Print model parameters
    print_model_parameters(model)

    # Create a random input tensor of shape [batch_size, channels, height, width]
    input_tensor = torch.randn(16, 1, 87, 128)  # Example: batch_size=16, channels=1, height=87, width=128

    # Forward pass through the model
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

# Run the test with parameters
if __name__ == "__main__":
    test_model_with_params()
