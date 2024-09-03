import pickle
import os
import torch
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import label, find_objects
from scipy.ndimage import convolve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import csv

class Evaluator(Dataset):
    def __init__(self, phase, dataset_paths, calibration_paths):
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

        self.calibration = False
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

        if self.calibration:
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

        spectrum_np = spectrum.numpy()
        pointcloud_gt_np = pointcloud.numpy()

        # Perform CFAR detection
        pointcloud_pred_np = self.apply_cfar(spectrum_np, self.current_cfar_type, self.current_threshold_factor)

        pointcloud_pred_np_mask = np.where(pointcloud_pred_np > 0, 1, 0)

        # Finding the two largest closures (bounding boxes) in the ground truth
        if 'lucacx' in os.path.basename(data_path):
            bboxes = self.find_two_largest_closure_bboxes(pointcloud_gt_np, h_expand=8)
        else:
            bboxes = [self.find_max_closure_bbox(pointcloud_gt_np, h_expand=8)]

        # Initialize variables for detection and false alarm rate calculation
        detected_cells = 0
        true_cells = 0
        falsealarm_cells = 0
        false_cells = 87 * 128  # Full figure size

        # Calculate the detection rate and false alarm rate
        for bbox in bboxes:
            if bbox:
                r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

                r1 = max(r1-8, 0)
                # Ensure r2 and c2 are within bounds
                r2 = min(r2+8, 86)  # r2 must be less than 87
                c2 = min(c2, 127)  # c2 must be less than 128

                bbox_height = r2 - r1
                bbox_width = c2 - c1

                # Add bounding box area (including boundaries) to true cells
                bbox_area = (bbox_height + 1) * (bbox_width + 1)
                true_cells += bbox_area
                false_cells -= bbox_area  # Remove the bounding box area from false cells

                for row in range(r1, r2 + 1):
                    row_points = pointcloud_pred_np_mask[row, c1:c2 + 1]
                    if np.any(row_points > 0):
                        detected_cells += bbox_area
                        break

        # Calculate false alarm cells
        outside_bbox_mask = np.ones_like(pointcloud_pred_np_mask)
        for bbox in bboxes:
            if bbox:
                r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

                r1 = max(r1-8, 0)
                # Ensure r2 and c2 are within bounds
                r2 = min(r2+8, 86)  # r2 must be less than 87
                c2 = min(c2, 127)  # c2 must be less than 128

                outside_bbox_mask[r1:r2 + 1, c1:c2 + 1] = 0  # Mask out bounding box areas

        falsealarm_cells = np.sum(pointcloud_pred_np_mask * outside_bbox_mask)

        # Update metrics
        self.total_detection_cells += detected_cells
        self.total_falsealarm_cells += falsealarm_cells
        self.total_true_cells += true_cells
        self.total_false_cells += false_cells

        # logger.debug(f"Detected Cells: {self.total_detection_cells}")
        # logger.debug(f"True Cells: {self.total_true_cells}")
        # logger.debug(f"False Alarm Cells: {self.total_falsealarm_cells}")
        # logger.debug(f"False Cells: {self.total_false_cells}")

        # logger.debug(f"Detection Rate: {self.total_detection_cells / self.total_true_cells}")
        # logger.debug(f"False Alarm Rate: {self.total_falsealarm_cells / self.total_false_cells}")

        isVisualizations = False  # Set to True to enable visualizations
        # Visualization code
        if isVisualizations:
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

                    r1 = max(r1-8, 0)
                    # Ensure r2 and c2 are within bounds
                    r2 = min(r2+8, 86)  # r2 must be less than 87
                    c2 = min(c2, 127)  # c2 must be less than 128

                    rect = patches.Rectangle((c1, r1), c2 - c1, r2 - r1, linewidth=2, edgecolor=colors[i], facecolor='none')
                    axes[2].add_patch(rect)

            plt.show()

            # Save the figure in the corresponding path under /evaluation_visualization/
            save_path = data_path.replace('/dataset/', '/cfar_evaluation_visualization/'+self.current_cfar_type+'/').replace('.pickle', '.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)

            logger.info(f'Visualization saved to {save_path}')

            plt.close(fig)  # Close the figure to free memory

        return 0
    
    def apply_cfar(self, spectrum, cfar_type='CA', threshold_factor=5.8):
        # CFAR Parameters
        guard_len = 3
        noise_len = 10

        if cfar_type == 'CA':
            return self.ca_cfar_2d(spectrum, guard_len=guard_len, noise_len=noise_len, threshold_factor=threshold_factor)
        elif cfar_type == 'OS':
            return self.os_cfar_2d(spectrum, guard_len=guard_len, noise_len=noise_len, threshold_factor=threshold_factor)
        elif cfar_type == 'GO':
            return self.go_cfar_2d(spectrum, guard_len=guard_len, noise_len=noise_len, threshold_factor=threshold_factor)
        elif cfar_type == 'SO':
            return self.so_cfar_2d(spectrum, guard_len=guard_len, noise_len=noise_len, threshold_factor=threshold_factor)
        else:
            raise ValueError(f"Unknown CFAR type: {cfar_type}")

    def ca_cfar_2d(self, x, guard_len=4, noise_len=8, threshold_factor=4, mode='reflect'):
        # Create a 2D kernel
        kernel_size = 1 + (2 * guard_len) + (2 * noise_len)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (2 * noise_len)**2
        kernel[noise_len:noise_len + (2 * guard_len) + 1, noise_len:noise_len + (2 * guard_len) + 1] = 0
        
        # Compute noise floor using 2D convolution
        noise_floor = convolve(x, kernel, mode=mode)
        
        # Calculate threshold
        threshold = threshold_factor * noise_floor
        
        # Apply CA-CFAR detection
        ret = (x > threshold)

        return ret

    def os_cfar_2d(self, x, guard_len=4, noise_len=8, threshold_factor=4, mode='reflect'):
        kernel_size = 1 + (2 * guard_len) + (2 * noise_len)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel[noise_len:noise_len + (2 * guard_len) + 1, noise_len:noise_len + (2 * guard_len) + 1] = 0
        
        # Pad the input for boundary conditions
        padded_x = np.pad(x, pad_width=noise_len + guard_len, mode=mode)
        
        noise_floor = np.zeros_like(x)
        
        for i in range(noise_floor.shape[0]):
            for j in range(noise_floor.shape[1]):
                region = padded_x[i:i + kernel_size, j:j + kernel_size]
                sorted_region = np.sort(region[kernel == 1].flatten())
                noise_floor[i, j] = sorted_region[len(sorted_region) // 2]  # Median
        
        threshold = threshold_factor * noise_floor
        
        ret = (x > threshold)
        
        return ret

    def go_cfar_2d(self, x, guard_len=4, noise_len=8, threshold_factor=4, mode='reflect'):
        kernel_size = 1 + (2 * guard_len) + (2 * noise_len)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel[noise_len:noise_len + (2 * guard_len) + 1, noise_len:noise_len + (2 * guard_len) + 1] = 0
        
        # Pad the input for boundary conditions
        padded_x = np.pad(x, pad_width=noise_len + guard_len, mode=mode)
        
        noise_floor = np.zeros_like(x)
        
        for i in range(noise_floor.shape[0]):
            for j in range(noise_floor.shape[1]):
                region = padded_x[i:i + kernel_size, j:j + kernel_size]
                leading_region = region[:, :noise_len]
                lagging_region = region[:, -noise_len:]
                noise_floor[i, j] = max(np.mean(leading_region), np.mean(lagging_region))
        
        threshold = threshold_factor * noise_floor
        
        ret = (x > threshold)
        
        return ret

    def so_cfar_2d(self, x, guard_len=4, noise_len=8, threshold_factor=4, mode='reflect'):
        kernel_size = 1 + (2 * guard_len) + (2 * noise_len)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel[noise_len:noise_len + (2 * guard_len) + 1, noise_len:noise_len + (2 * guard_len) + 1] = 0
        
        # Pad the input for boundary conditions
        padded_x = np.pad(x, pad_width=noise_len + guard_len, mode=mode)
        
        noise_floor = np.zeros_like(x)
        
        for i in range(noise_floor.shape[0]):
            for j in range(noise_floor.shape[1]):
                region = padded_x[i:i + kernel_size, j:j + kernel_size]
                leading_region = region[:, :noise_len]
                lagging_region = region[:, -noise_len:]
                noise_floor[i, j] = min(np.mean(leading_region), np.mean(lagging_region))
        
        threshold = threshold_factor * noise_floor
        
        ret = (x > threshold)
        
        return ret

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

    def find_max_closure_bbox(self, matrix, h_expand=8):
        # Label connected components of 1s
        labeled_array, num_features = label(matrix)
        
        # Find bounding boxes of labeled components
        bounding_boxes = find_objects(labeled_array)
        
        if not bounding_boxes:  # If no bounding boxes found, return None or an empty list
            return []

        # Determine the largest connected component based on area
        max_area = 0
        max_bbox = None
        for bbox in bounding_boxes:
            if bbox is not None:
                area = (bbox[0].stop - bbox[0].start) * (bbox[1].stop - bbox[1].start)
                if area > max_area:
                    # max_bbox = bbox
                    max_bbox = bbox

        if max_bbox is None:
            return []  # Return an empty list if no bounding box is found
        
        r1, c1, r2, c2 = max_bbox[0].start, max_bbox[1].start, max_bbox[0].stop, max_bbox[1].stop

        # Extend r1 and r2 to the nearest multiple of 15.
        r1_extended = max(0, r1 - (r1 % 15))
        r2_extended = min(86, r2 + (15 - (r2 % 15)))  # Ensure r2 stays within bounds

        # Return the extended bounding box
        return [slice(r1_extended, r2_extended), slice(c1, c2)]

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
    
    # Initialize dataset and dataloader
    eval_dataset = Evaluator(phase='test', dataset_paths=dataset_paths, calibration_paths=calibration_paths)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # CSV file to save results
    results_file = 'cfar_evaluation_results.csv'
    
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['CFAR_Type', 'Threshold_Factor', 'Detected_Cells', 'True_Cells', 'False_Alarm_Cells', 'False_Cells', 'Detection_Rate', 'False_Alarm_Rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        cfar_types = ['CA', 'OS', 'GO', 'SO']
        guard_len = 3
        noise_len = 10

        for cfar_type in cfar_types:
            results_file = f'cfar_evaluation_results_{cfar_type}.csv'
            
            with open(results_file, 'w', newline='') as csvfile:
                fieldnames = ['Threshold_Factor', 'Detected_Cells', 'True_Cells', 'False_Alarm_Cells', 'False_Cells', 'Detection_Rate', 'False_Alarm_Rate']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for threshold_factor in np.arange(0, 20.05, 0.05):
                    # Reset metrics for each new threshold factor
                    eval_dataset.total_detection_cells = 0
                    eval_dataset.total_falsealarm_cells = 0
                    eval_dataset.total_true_cells = 0
                    eval_dataset.total_false_cells = 0

                    eval_dataset.current_cfar_type = cfar_type
                    eval_dataset.current_threshold_factor = threshold_factor
                    
                    for data in tqdm(eval_loader, desc=f'Processing CFAR: {cfar_type} with Threshold: {threshold_factor}'):
                        eval_dataset.__getitem__(0)  # Run processing

                    # Save results to CSV
                    writer.writerow({
                        'Threshold_Factor': threshold_factor,
                        'Detected_Cells': eval_dataset.total_detection_cells,
                        'True_Cells': eval_dataset.total_true_cells,
                        'False_Alarm_Cells': eval_dataset.total_falsealarm_cells,
                        'False_Cells': eval_dataset.total_false_cells,
                        'Detection_Rate': eval_dataset.total_detection_cells / eval_dataset.total_true_cells if eval_dataset.total_true_cells > 0 else 0,
                        'False_Alarm_Rate': eval_dataset.total_falsealarm_cells / eval_dataset.total_false_cells if eval_dataset.total_false_cells > 0 else 0
                    })

            logger.info(f'CFAR evaluation results for {cfar_type} saved to {results_file}')

