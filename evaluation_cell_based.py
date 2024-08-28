import pickle
import os
import torch
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import label, find_objects
from torch.utils.data import DataLoader, Dataset
from model import LSSLModel
from args import get_args
from tqdm import tqdm
from datetime import datetime

class Evaluator(Dataset):
    def __init__(self, phase, dataset_paths, calibration_paths, checkpoint_path):
        assert len(dataset_paths) == len(calibration_paths), "Each dataset path must have a corresponding calibration path"
        
        self.phase = phase
        self.data_list = []
        self.dataset_paths = dataset_paths
        self.calibration_paths = calibration_paths
        self.calibration_spectrums = []

        # Load data paths
        for dataset_path in self.dataset_paths:
            data_path = os.path.join(dataset_path, phase)
            self.data_list += [os.path.join(data_path, data) for data in os.listdir(data_path)]

        self.calibration = True
        if self.calibration:
            logger.info('Calibration mode enabled')
            for calibration_path in self.calibration_paths:
                calibration_files = os.listdir(calibration_path)
                calibration_spectrum = torch.zeros(87, 128)
                for calibration_file in calibration_files:
                    with open(os.path.join(calibration_path, calibration_file), 'rb') as f:
                        calibration_data = pickle.load(f)
                    calibration_data = calibration_data['spectrum']
                    calibration_data = torch.tensor(calibration_data, dtype=torch.float32).flip(0)
                    calibration_data = calibration_data[:, 14:]
                    calibration_spectrum += torch.tensor(calibration_data, dtype=torch.float32)
                calibration_spectrum = calibration_spectrum / len(calibration_files)
                self.calibration_spectrums.append(calibration_spectrum)

        # Load the model
        args = get_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = LSSLModel(
            num_layers=4,
            d=256,
            order=256,
            dt_min=1e-3,
            dt_max=8e-5,
            channels=1,
            dropout=0.1
        ).to(self.device)

        model.load_state_dict(torch.load(checkpoint_path))
        self.model = model.eval()

        # Initialize evaluation metrics
        self.total_detection_cells = 0
        self.total_falsealarm_cells = 0

        self.total_true_cells = 0
        self.total_false_cells = 0

    def __len__(self):
        logger.info(f'Dataset Loaded, Size: {len(self.data_list)}')
        return len(self.data_list)

    def __getitem__(self, idx):
        data_path = self.data_list[idx]

        # Determine which dataset and corresponding calibration spectrum to use
        for i, dataset_path in enumerate(self.dataset_paths):
            if data_path.startswith(dataset_path):
                calibration_spectrum = self.calibration_spectrums[i]
                break
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        spectrum = torch.tensor(data['spectrum'], dtype=torch.float32)
        pointcloud = torch.tensor(data['pointcloud'], dtype=torch.float32)
        spectrum = spectrum[:, 14:]
        pointcloud = pointcloud[:, 14:]

        if self.calibration:
            spectrum = spectrum - calibration_spectrum

            pointcloud = torch.where(pointcloud == 1, torch.tensor(0), pointcloud)
            pointcloud = torch.where(pointcloud == 2, torch.tensor(1), pointcloud)

        # Flatten the spectrum and pointcloud
        spectrum = spectrum.flatten()
        pointcloud = pointcloud.flatten()

        with torch.no_grad():
            spectrum_input = spectrum.unsqueeze(0).to(self.device)
            pointcloud_pred = self.model(spectrum_input)

        pointcloud_pred_np = pointcloud_pred.squeeze().detach().cpu().view(87, 128).numpy()
        pointcloud_gt_np = pointcloud.view(87, 128).numpy()
        spectrum_np = spectrum.view(87, 128).numpy()

        # Finding the two largest closures (bounding boxes) in the ground truth
        bboxes = self.find_two_largest_closure_bboxes(pointcloud_gt_np, h_expand=8)
        pointcloud_pred_np_mask = np.where(pointcloud_pred_np > 0, 1, 0)
        non_zero_positions = np.argwhere(pointcloud_pred_np_mask == 1)

        # Initialize variables for detection and false alarm rate calculation
        detected_cells = 0
        true_cells = 0
        falsealarm_cells = 0
        false_cells = 87 * 128  # Full figure size

        # Calculate the detection rate and false alarm rate
        for bbox in bboxes:
            if bbox:
                r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

                # Ensure r2 and c2 are within bounds
                r2 = min(r2, 86)  # r2 must be less than 87
                c2 = min(c2, 127)  # c2 must be less than 128

                bbox_height = r2 - r1
                bbox_width = c2 - c1

                # Add bounding box area (including boundaries) to true cells
                bbox_area = (bbox_height + 1) * (bbox_width + 1)
                true_cells += bbox_area
                false_cells -= bbox_area  # Remove the bounding box area from false cells

                # Check each row within the bounding box, if any point in the bbox
                # the entire bbox area is considered detected
                

                for row in range(r1, r2 + 1):
                    row_points = pointcloud_pred_np_mask[row, c1:c2 + 1]
                    if np.any(row_points > 0):
                        detected_cells += bbox_area
                        break
                #         detected_cells += bbox_width + 1  # Add the entire row width of the bounding box to detected cells

        # Calculate false alarm cells
        outside_bbox_mask = np.ones_like(pointcloud_pred_np_mask)
        for bbox in bboxes:
            if bbox:
                r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

                # Ensure r2 and c2 are within bounds
                r2 = min(r2, 86)  # r2 must be less than 87
                c2 = min(c2, 127)  # c2 must be less than 128

                outside_bbox_mask[r1:r2 + 1, c1:c2 + 1] = 0  # Mask out bounding box areas

        falsealarm_cells = np.sum(pointcloud_pred_np_mask * outside_bbox_mask)

        # Update metrics
        self.total_detection_cells += detected_cells
        self.total_falsealarm_cells += falsealarm_cells
        self.total_true_cells += true_cells
        self.total_false_cells += false_cells

        logger.debug(f"Detected Cells: {self.total_detection_cells}")
        logger.debug(f"True Cells: {self.total_true_cells}")
        logger.debug(f"False Alarm Cells: {self.total_falsealarm_cells}")
        logger.debug(f"False Cells: {self.total_false_cells}")

        # Visualization code
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(spectrum_np, cmap='viridis', aspect='auto')
        axes[0].set_title('Spectrum')

        axes[1].imshow(pointcloud_pred_np_mask, cmap='gray', aspect='auto')
        axes[1].set_title('Predicted Pointcloud Mask')

        axes[2].imshow(pointcloud_gt_np, cmap='gray', aspect='auto')
        axes[2].set_title('Ground Truth Pointcloud')

        # Draw bounding boxes
        colors = ['red', 'green']
        for i, bbox in enumerate(bboxes):
            if bbox:
                r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

                # Ensure r2 and c2 are within bounds
                r2 = min(r2, 86)  # r2 must be less than 87
                c2 = min(c2, 127)  # c2 must be less than 128

                rect = patches.Rectangle((c1, r1), c2 - c1, r2 - r1, linewidth=2, edgecolor=colors[i], facecolor='none')
                axes[2].add_patch(rect)

        # Draw all non-zero positions
        for pos in non_zero_positions:
            axes[2].plot(pos[1], pos[0], 'bo', markersize=5)

        # Add a legend to indicate the points and bounding boxes
        axes[2].legend(['Points (Inside BBox)', 'Points (Outside BBox)', 'Largest BBox', 'Second Largest BBox'])

        plt.show()

        # Save the figure in the corresponding path under /evaluation_visualization/
        save_path = data_path.replace('/dataset/', '/evaluation_visualization/').replace('.pickle', '.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

        logger.info(f'Visualization saved to {save_path}')

        plt.close(fig)  # Close the figure to free memory

        return 0

    def find_two_largest_closure_bboxes(self, matrix, h_expand=0):
        labeled_array, num_features = label(matrix)
        bounding_boxes = find_objects(labeled_array)
        # bounding_boxes are sorted by size in descending order.
        bounding_boxes = sorted(bounding_boxes, key=lambda bbox: (bbox[0].stop - bbox[0].start) * (bbox[1].stop - bbox[1].start), reverse=True)

        expanded_bboxes = []
        for bbox in bounding_boxes[:2]:
            r1, c1 = bbox[0].start, bbox[1].start
            r2, c2 = bbox[0].stop, bbox[1].stop 
            r1 = max(r1 - h_expand, 0)
            r2 = min(r2 + h_expand, matrix.shape[0] - 1)
            expanded_bboxes.append((slice(r1, r2), slice(c1, c2)))

        return expanded_bboxes

if __name__ == '__main__':
    dataset_paths = [
        "/data/lucayu/lss-cfar/dataset/cx_corridor_2024-08-27",
        "/data/lucayu/lss-cfar/dataset/cx_env_corridor_2024-08-27",
        "/data/lucayu/lss-cfar/dataset/luca_env_hw_101_2024-08-23",
        "/data/lucayu/lss-cfar/dataset/luca_hw_101_2024-08-23",
        "/data/lucayu/lss-cfar/dataset/lucacx_corridor_2024-08-27",
        "/data/lucayu/lss-cfar/dataset/lucacx_env_corridor_2024-08-27",
        "/data/lucayu/lss-cfar/dataset/wayne_env_office_2024-08-27",
        "/data/lucayu/lss-cfar/dataset/wayne_office_2024-08-27"
    ]
    calibration_paths = [
        "/data/lucayu/lss-cfar/raw_dataset/cx_env_corridor_2024-08-27",
        "/data/lucayu/lss-cfar/raw_dataset/cx_env_corridor_2024-08-27",
        "/data/lucayu/lss-cfar/raw_dataset/luca_env_hw_101_2024-08-23",
        "/data/lucayu/lss-cfar/raw_dataset/luca_env_hw_101_2024-08-23",
        "/data/lucayu/lss-cfar/raw_dataset/lucacx_env_corridor_2024-08-27",
        "/data/lucayu/lss-cfar/raw_dataset/lucacx_env_corridor_2024-08-27",
        "/data/lucayu/lss-cfar/raw_dataset/wayne_env_office_2024-08-27",
        "/data/lucayu/lss-cfar/raw_dataset/wayne_env_office_2024-08-27"
    ]
    
    checkpoint_path = '/home/lucayu/lss-cfar/checkpoints/20240828-200315_model_layers_4_hidden_256_order_256_dtmin_0.001_dtmax_8e-05_channels_1_dropout_0.0_lr_0.01_batch_4_steps_10000_optimizer_AdamW_decay_0.1_step_300_gamma_0.5_losstype_l1/20240828-200315_model_layers_4_hidden_256_order_256_dtmin_0.001_dtmax_8e-05_channels_1_dropout_0.0_lr_0.01_batch_4_steps_10000_optimizer_AdamW_decay_0.1_step_300_gamma_0.5_losstype_l1.pt'
    eval_dataset = Evaluator(phase='test', dataset_paths=dataset_paths, calibration_paths=calibration_paths, checkpoint_path=checkpoint_path)

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    for data in tqdm(eval_loader):
        pass  # The __getitem__ method handles processing and accumulation

