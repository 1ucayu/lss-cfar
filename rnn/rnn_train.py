import os
import time
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import datetime

from rnn_model import CFARRNN  # Importing the combined model
from rnn_dataset import RNNDataset  # Importing the dataset
from rnn_args import get_args  # Importing the argument parser

def generate_checkpoint_name(args):
    """Generate a descriptive checkpoint name based on the training arguments, including a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Format: YYYYMMDD-HHMMSS
    return (
        f"rnn_cnn_"
        f"{timestamp}_"
        f"layers_{args.num_layers}_hidden_{args.hidden_size}_lr_{args.learning_rate}_batch_{args.batch_size}_steps_{args.total_steps}_"
        f"optimizer_{args.optimizer}_decay_{args.weight_decay}_step_{args.step_size}_gamma_{args.gamma}"
    )

def visualize_result(spectrum, pointcloud_pred, pointcloud_gt, writer, step):
    idx = np.random.randint(0, spectrum.shape[0])  # Randomly select one from the batch
    spectrum_np = spectrum[idx].cpu().numpy().squeeze()
    pointcloud_pred_np = pointcloud_pred[idx].detach().cpu().numpy().squeeze()
    pointcloud_gt_np = pointcloud_gt[idx].cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(spectrum_np, cmap='viridis', aspect='auto')
    axes[0].set_title('Spectrum')
    
    axes[1].imshow(pointcloud_pred_np, cmap='viridis', aspect='auto')
    axes[1].set_title('Predicted Pointcloud')
    
    axes[2].imshow(pointcloud_gt_np, cmap='gray', aspect='auto')
    axes[2].set_title('Ground Truth Pointcloud')

    writer.add_figure('Visualization', fig, global_step=step)
    plt.close(fig)

def train():
    args = get_args()
    
    # Set GPU devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate the checkpoint name
    checkpoint_name = generate_checkpoint_name(args)
    checkpoint_dir = os.path.join(args.save_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup TensorBoard and checkpoint directory
    log_dir = os.path.join(args.log_dir, checkpoint_name)
    writer = SummaryWriter(log_dir=log_dir)

    scalar = torch.cuda.amp.GradScaler()

    # Create model
    model = CFARRNN().to(device)
    
    # Load dataset
    train_dataset = RNNDataset(phase='train', dataset_paths=args.dataset_paths, calibration_paths=args.calibration_paths)
    val_dataset = RNNDataset(phase='test', dataset_paths=args.dataset_paths, calibration_paths=args.calibration_paths)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset._collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=val_dataset._collate_fn)

    # Loss and optimizer
    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_loss = float('inf')
    current_step = 0

    while current_step < args.total_steps:
        for phase, loader in [('train', train_loader), ('test', val_loader)]:
            if phase == 'train':
                model.train()  # Switch to training mode
            else:
                model.eval()
                accumulated_test_loss = 0.0
                num_test_steps = 0

            for batch in loader:
                spectrum = batch['spectrum'].unsqueeze(1).to(device)  # Add channel dimension [b, 1, 87, 128]
                pointcloud_gt = batch['pointcloud'].unsqueeze(1).to(device)  # Add channel dimension [b, 1, 87, 128]

                if phase == 'train':
                    with torch.cuda.amp.autocast():
                        pointcloud_pred = model(spectrum)
                        loss = criterion(pointcloud_pred, pointcloud_gt)

                    optimizer.zero_grad()
                    scalar.scale(loss).backward()  # Scale loss for mixed precision training
                    scalar.step(optimizer)
                    scalar.update()

                    # Log the current learning rate every step
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Learning Rate', current_lr, current_step)

                    # Step the scheduler every iteration if needed
                    scheduler.step()

                    logger.info(f"Step {current_step}/{args.total_steps}, Loss: {loss.item():.6f}")
                    writer.add_scalar(f'Loss/{phase}', loss.item(), current_step)

                    if current_step % args.visualization_stride == 0:
                        visualize_result(spectrum, pointcloud_pred, pointcloud_gt, writer, current_step)

                    current_step += 1

                    if current_step >= args.total_steps:
                        break
                else:
                    with torch.no_grad():
                        pointcloud_pred = model(spectrum)
                        loss = criterion(pointcloud_pred, pointcloud_gt)
                        accumulated_test_loss += loss.item()
                        num_test_steps += 1

            if phase == 'test':
                average_test_loss = accumulated_test_loss / num_test_steps
                writer.add_scalar('Loss/test', average_test_loss, current_step)

                if average_test_loss < best_loss:
                    best_loss = average_test_loss
                    save_path = os.path.join(checkpoint_dir, checkpoint_name + ".pt")
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Best model saved with test loss: {best_loss:.6f}")

            scheduler.step()

if __name__ == "__main__":
    train()
