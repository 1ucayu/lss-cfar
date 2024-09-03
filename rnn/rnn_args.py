import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RNN Model Training')
    
    # Dataset hyperparameters
    parser.add_argument('--dataset_paths', type=str, nargs='+', help='List of dataset paths')
    parser.add_argument('--calibration_paths', type=str, nargs='+', help='List of calibration paths')
    
    # Model hyperparameters
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the RNN')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size of the RNN')
    parser.add_argument('--input_size', type=int, default=87*128, help='Input size to the RNN (87*128)')
    parser.add_argument('--output_size', type=int, default=87*128, help='Output size of the RNN (87*128)')
    
    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--total_steps', type=int, default=10000, help='Total number of training steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--step_size', type=int, default=1000, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Scheduler gamma')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--loss_type', type=str, default='bce', help='Loss function type (e.g., bce, mse)')
    
    # Other configurations
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use, e.g., "0,1"')
    parser.add_argument('--visualization_stride', type=int, default=100, help='Steps between visualizations')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')

    args = parser.parse_args()
    return args
