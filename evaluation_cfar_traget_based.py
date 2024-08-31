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
    def __init__(self, phase, dataset_paths, calibration_paths):
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

        self.total_true_positives = 0
        self.total_false_positives = 0
        self.total_gt_positives = 0
        self.total_predicted_positives = 0

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

        spectrum_np = spectrum.view(87, 128).numpy()
        pointcloud_gt_np = pointcloud.view(87, 128).numpy()

        # Perform CFAR detection
        cfar_type = 'CA'  # Change this to 'OS', 'GO', or 'SO' to test different CFAR methods
        pointcloud_pred_np = self.apply_cfar(spectrum_np, cfar_type)

        # Get the positions marked by CFAR
        detected_positions = np.argwhere(pointcloud_pred_np == 1)

        # Calculate the centroid of the detected positions
        if detected_positions.size > 0:
            centroid = np.mean(detected_positions, axis=0).astype(int)
            logger.info(f"Centroid of detected positions: {centroid}")
        else:
            centroid = None
            logger.info("No positions detected by CFAR.")

        # Finding the bounding box of the maximum closure of 1s in pointcloud_gt_np
        bbox = self.find_max_closure_bbox(pointcloud_gt_np)
        
        # Extend the bounding box based on azimuth intervals
        if bbox:
            bbox = self.extend_bbox_to_azimuth_intervals(bbox)

        # Check if the centroid is inside the extended ground truth bounding box
        inside_bbox = False
        if centroid is not None and bbox:
            r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop
            inside_bbox = (r1 <= centroid[0] < r2) and (c1 <= centroid[1] < c2)
            logger.info(f'Is the centroid inside the bounding box? {"Yes" if inside_bbox else "No"}')
        else:
            logger.info("No bounding box found or no centroid available.")

        # Accumulate the statistics for the full dataset
        self.total_gt_positives += np.sum(pointcloud_gt_np == 1)  # Ground truth positives
        if inside_bbox:
            self.total_true_positives += 1  # True positive if centroid is inside the bbox
        else:
            self.total_false_positives += 1  # False positive if centroid is outside the bbox

        self.total_predicted_positives += 1  # Count this prediction for false alarm calculation

        # Visualization code (unchanged)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(spectrum_np, cmap='viridis', aspect='auto')
        axes[0].set_title('Spectrum')

        axes[1].imshow(pointcloud_pred_np, cmap='viridis', aspect='auto')
        axes[1].set_title('Predicted Pointcloud')

        axes[2].imshow(pointcloud_gt_np, cmap='gray', aspect='auto')
        axes[2].set_title('Ground Truth Pointcloud')

        # If a bounding box was found, draw it on the Ground Truth Pointcloud
        if bbox:
            r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop
            rect = patches.Rectangle((c1, r1), c2 - c1, r2 - r1, linewidth=2, edgecolor='red', facecolor='none')
            axes[2].add_patch(rect)

        # Draw the centroid position if detected
        if centroid is not None:
            axes[2].plot(centroid[1], centroid[0], 'bo', markersize=8, label='Centroid')

        # Add a legend to indicate the centroid
        axes[2].legend()

        # Save the figure to the given folder path with an automatic name
        save_folder = os.path.join('/data/lucayu/lss-cfar/', cfar_type + '_evaluation_visualization')
        os.makedirs(save_folder, exist_ok=True)
        file_name = f'sample_{idx:04d}.png'
        save_path = os.path.join(save_folder, file_name)
        fig.savefig(save_path)

        logger.info(f'Visualization saved to {save_path}')

        plt.close(fig)  # Close the figure to free memory

        return 0
    
    def apply_cfar(self, spectrum, cfar_type='OS'):
        # CFAR Parameters
        guard_cells = 5
        training_cells = 30
        threshold_factor = 10

        # Apply CFAR based on the specified method
        if cfar_type == 'CA':
            return self.ca_cfar(spectrum, guard_cells, training_cells, threshold_factor)
        elif cfar_type == 'OS':
            return self.os_cfar(spectrum, guard_cells, training_cells, threshold_factor)
        elif cfar_type == 'GO':
            return self.go_cfar(spectrum, guard_cells, training_cells, threshold_factor)
        elif cfar_type == 'SO':
            return self.so_cfar(spectrum, guard_cells, training_cells, threshold_factor)
        else:
            raise ValueError(f"Unknown CFAR type: {cfar_type}")

    def ca_cfar(self, spectrum, guard_cells, training_cells, threshold_factor):
        # Implement Cell Averaging CFAR
        cfar_output = np.zeros_like(spectrum)
        rows, cols = spectrum.shape

        for i in range(rows):
            for j in range(cols):
                if j < training_cells + guard_cells or j > cols - (training_cells + guard_cells):
                    continue  # Skip edges

                leading_cells = spectrum[i, j - training_cells - guard_cells:j - guard_cells]
                lagging_cells = spectrum[i, j + guard_cells + 1:j + guard_cells + training_cells + 1]

                noise_estimate = np.mean(np.concatenate((leading_cells, lagging_cells)))
                threshold = noise_estimate * threshold_factor

                if spectrum[i, j] > threshold:
                    cfar_output[i, j] = 1

        return cfar_output

    def os_cfar(self, spectrum, guard_cells, training_cells, threshold_factor):
        # Implement Ordered Statistic CFAR
        cfar_output = np.zeros_like(spectrum)
        rows, cols = spectrum.shape

        for i in range(rows):
            for j in range(cols):
                if j < training_cells + guard_cells or j > cols - (training_cells + guard_cells):
                    continue  # Skip edges

                leading_cells = spectrum[i, j - training_cells - guard_cells:j - guard_cells]
                lagging_cells = spectrum[i, j + guard_cells + 1:j + guard_cells + training_cells + 1]

                noise_samples = np.concatenate((leading_cells, lagging_cells))
                sorted_noise = np.sort(noise_samples)
                noise_estimate = sorted_noise[training_cells // 2]  # Use the median value

                threshold = noise_estimate * threshold_factor

                if spectrum[i, j] > threshold:
                    cfar_output[i, j] = 1

        return cfar_output

    def go_cfar(self, spectrum, guard_cells, training_cells, threshold_factor):
        # Implement Greatest Of CFAR
        cfar_output = np.zeros_like(spectrum)
        rows, cols = spectrum.shape

        for i in range(rows):
            for j in range(cols):
                if j < training_cells + guard_cells or j > cols - (training_cells + guard_cells):
                    continue  # Skip edges

                leading_cells = spectrum[i, j - training_cells - guard_cells:j - guard_cells]
                lagging_cells = spectrum[i, j + guard_cells + 1:j + guard_cells + training_cells + 1]

                noise_estimate = max(np.mean(leading_cells), np.mean(lagging_cells))
                threshold = noise_estimate * threshold_factor

                if spectrum[i, j] > threshold:
                    cfar_output[i, j] = 1

        return cfar_output

    def so_cfar(self, spectrum, guard_cells, training_cells, threshold_factor):
        # Implement Smallest Of CFAR
        cfar_output = np.zeros_like(spectrum)
        rows, cols = spectrum.shape

        for i in range(rows):
            for j in range(cols):
                if j < training_cells + guard_cells or j > cols - (training_cells + guard_cells):
                    continue  # Skip edges

                leading_cells = spectrum[i, j - training_cells - guard_cells:j - guard_cells]
                lagging_cells = spectrum[i, j + guard_cells + 1:j + guard_cells + training_cells + 1]

                noise_estimate = min(np.mean(leading_cells), np.mean(lagging_cells))
                threshold = noise_estimate * threshold_factor

                if spectrum[i, j] > threshold:
                    cfar_output[i, j] = 1

        return cfar_output

    def extend_bbox_to_azimuth_intervals(self, bbox):
        # Original bounding box coordinates
        r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

        # The resolution for azimuth is 15 bins, so we extend the bounding box accordingly.
        # Extend r1 and r2 to the nearest multiple of 15.
        r1_extended = max(0, r1 - (r1 % 15))
        r2_extended = min(86, r2 + (15 - (r2 % 15)))  # Ensure r2 stays within bounds

        # Return the extended bounding box
        return (slice(r1_extended, r2_extended), slice(c1, c2))

    def find_max_closure_bbox(self, matrix):
        # Label connected components of 1s
        labeled_array, num_features = label(matrix)
        
        # Find bounding boxes of labeled components
        bounding_boxes = find_objects(labeled_array)
        
        # Determine the largest connected component based on area
        max_area = 0
        max_bbox = None
        for bbox in bounding_boxes:
            if bbox is not None:
                area = (bbox[0].stop - bbox[0].start) * (bbox[1].stop - bbox[1].start)
                if area > max_area:
                    max_area = area
                    max_bbox = bbox

        return max_bbox
    
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
        # Label connected components of 1s
        labeled_array, num_features = label(matrix)
        
        # Find bounding boxes of labeled components
        bounding_boxes = find_objects(labeled_array)
        
        # Determine the largest connected component based on area
        max_area = 0
        max_bbox = None
        for bbox in bounding_boxes:
            if bbox is not None:
                area = (bbox[0].stop - bbox[0].start) * (bbox[1].stop - bbox[1].start)
                if area > max_area:
                    max_area = area
                    max_bbox = bbox
        expanded_bboxes = []
        for bbox in bounding_boxes[:2]:
            r1, c1 = bbox[0].start, bbox[1].start
            r2, c2 = bbox[0].stop, bbox[1].stop 
            r1 = max(r1 - h_expand, 0)
            r2 = min(r2 + h_expand, matrix.shape[0] - 1)
            expanded_bboxes.append((slice(r1, r2), slice(c1, c2)))

        return max_bbox

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
    
    eval_dataset = Evaluator(phase='test', dataset_paths=dataset_paths, calibration_paths=calibration_paths)

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    for data in tqdm(eval_loader):
        pass  # The __getitem__ method handles processing and accumulation

