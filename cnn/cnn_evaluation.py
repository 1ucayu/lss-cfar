import pickle
import os
import torch
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import label, find_objects
from torch.utils.data import DataLoader, Dataset
from cnn_model import CFARCNN  # Import your CNN model
from cnn_args import get_args
from tqdm import tqdm

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

        model = CFARCNN().to(self.device)  # Use the CNN model

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

        pointcloud = pointcloud.unsqueeze(0)
        spectrum = spectrum.unsqueeze(0)

        pointcloud = torch.where(pointcloud == 1, torch.tensor(0), pointcloud)
        pointcloud = torch.where(pointcloud == 2, torch.tensor(1), pointcloud)
        
        # Normalize the spectrum as done during training (min-max normalization)
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())

        

        with torch.no_grad():
            spectrum_input = spectrum.unsqueeze(0).to(self.device)
            pointcloud_pred = self.model(spectrum_input)


        pointcloud_pred_np = pointcloud_pred.squeeze().detach().cpu().numpy()
        pointcloud_gt_np = pointcloud.cpu().numpy().squeeze()
        spectrum_np = spectrum.cpu().numpy().squeeze()

        # Finding the two largest closures (bounding boxes) in the ground truth
        if 'lucacx' in os.path.basename(data_path):
            bboxes = self.find_two_largest_closure_bboxes(pointcloud_gt_np, h_expand=8)
        else:
            bboxes = [self.find_max_closure_bbox(pointcloud_gt_np, h_expand=8)]

        pointcloud_pred_np_mask = np.where(pointcloud_pred_np > 0.025, 1, 0)
        non_zero_positions = np.argwhere(pointcloud_pred_np_mask == 1)

        # Initialize variables for detection and false alarm rate calculation
        detected_cells = 0
        true_cells = 0
        falsealarm_cells = 0
        false_cells = 87 * 128  # Full figure size

        for bbox in bboxes:
            if bbox:
                r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

                r2 = min(r2, 86)
                c2 = min(c2, 127)

                bbox_area = (r2 - r1 + 1) * (c2 - c1 + 1)
                true_cells += bbox_area
                false_cells -= bbox_area

                for row in range(r1, r2 + 1):
                    row_points = pointcloud_pred_np_mask[row, c1:c2 + 1]
                    if np.any(row_points > 0):
                        detected_cells += bbox_area
                        break

        outside_bbox_mask = np.ones_like(pointcloud_pred_np_mask)
        for bbox in bboxes:
            if bbox:
                r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

                r2 = min(r2, 86)
                c2 = min(c2, 127)

                outside_bbox_mask[r1:r2 + 1, c1:c2 + 1] = 0

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

        logger.debug(f"Detection Rate: {self.total_detection_cells / self.total_true_cells}")
        logger.debug(f"False Alarm Rate: {self.total_falsealarm_cells / self.total_false_cells}")

        # Visualization code
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(spectrum_np, cmap='viridis', aspect='auto')
        axes[0].set_title('Spectrum')

        axes[1].imshow(pointcloud_pred_np_mask, cmap='gray', aspect='auto')
        axes[1].set_title('Predicted Pointcloud Mask')

        axes[2].imshow(pointcloud_gt_np, cmap='gray', aspect='auto')
        axes[2].set_title('Ground Truth Pointcloud')

        colors = ['red', 'green']
        for i, bbox in enumerate(bboxes):
            if bbox:
                r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

                r2 = min(r2, 86)
                c2 = min(c2, 127)

                rect = patches.Rectangle((c1, r1), c2 - c1, r2 - r1, linewidth=2, edgecolor=colors[i], facecolor='none')
                axes[2].add_patch(rect)

        for pos in non_zero_positions:
            axes[2].plot(pos[1], pos[0], 'bo', markersize=5)

        axes[2].legend(['Points (Inside BBox)', 'Points (Outside BBox)', 'Largest BBox', 'Second Largest BBox'])

        plt.show()

        save_path = data_path.replace('/dataset/', '/cnn_evaluation_visualization/').replace('.pickle', '.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

        logger.info(f'Visualization saved to {save_path}')

        plt.close(fig)

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

    def find_max_closure_bbox(self, matrix, h_expand=0):
        labeled_array, num_features = label(matrix)
        bounding_boxes = find_objects(labeled_array)

        max_area = 0
        max_bbox = None
        for bbox in bounding_boxes:
            if bbox is not None:
                area = (bbox[0].stop - bbox[0].start) * (bbox[1].stop - bbox[1].start)
                if area > max_area:
                    max_area = area
                    max_bbox = bbox

        if max_bbox:
            r1, c1 = max_bbox[0].start, max_bbox[1].start
            r2, c2 = max_bbox[0].stop, max_bbox[1].stop 
            r1 = max(r1 - h_expand, 0)
            r2 = min(r2 + h_expand, matrix.shape[0] - 1)
            return (slice(r1, r2), slice(c1, c2))
        return None

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
    
    checkpoint_path = '/home/lucayu/lss-cfar/cnn/checkpoints/cnn_20240903-165848_conv_layers_5_filters_64_lr_0.0001_batch_16_steps_10000_optimizer_AdamW_decay_0.01_step_1000_gamma_0.5/cnn_20240903-165848_conv_layers_5_filters_64_lr_0.0001_batch_16_steps_10000_optimizer_AdamW_decay_0.01_step_1000_gamma_0.5.pt'
    eval_dataset = Evaluator(phase='test', dataset_paths=dataset_paths, calibration_paths=calibration_paths, checkpoint_path=checkpoint_path)

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    for data in tqdm(eval_loader):
        pass  # The __getitem__ method handles processing and accumulation
