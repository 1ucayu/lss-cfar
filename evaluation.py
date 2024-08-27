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
from sklearn.cluster import DBSCAN

class evaluator(Dataset):

    def __init__(self, phase, dataset_path, checkpoint_path):
        self.phases = phase
        self.data_list = []
        self.dataset_path = dataset_path
        data_path = f'{dataset_path}/{phase}'
        self.data_list += [f'{dataset_path}/{phase}/{data}' for data in os.listdir(data_path)]
        self.calibration = True
        if self.calibration:
            logger.info('Calibration mode enabled')
            calibration_path = '/data/lucayu/lss-cfar/raw_dataset/lucacx_env_corridor_2024-08-27'
            calibration_files = os.listdir(calibration_path)
            calibration_spectrum = torch.zeros(87, 128)
            for calibration_file in calibration_files:
                with open(f'{calibration_path}/{calibration_file}', 'rb') as f:
                    calibration_data = pickle.load(f)
                calibration_data = calibration_data['spectrum']
                calibration_data = torch.tensor(calibration_data, dtype=torch.float32).flip(0)
                calibration_data = calibration_data[:,14:]
                calibration_spectrum += torch.tensor(calibration_data, dtype=torch.float32)
            calibration_spectrum = calibration_spectrum / len(calibration_files)
            self.calibration_spectrum = calibration_spectrum
        # load the model
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

        # load the model
        model.load_state_dict(torch.load(checkpoint_path))
        self.model = model.eval()

        self.total_true_positives = 0
        self.total_false_positives = 0
        self.total_gt_positives = 0
        self.total_predicted_positives = 0

    def __len__(self):
        logger.info(f'Dataset Loaded, Size: {len(self.data_list)}')
        return len(self.data_list)
    

    def __getitem__(self, idx):
        pickle_path = self.data_list[idx]
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        spectrum = torch.tensor(data['spectrum'], dtype=torch.float32)
        pointcloud = torch.tensor(data['pointcloud'], dtype=torch.float32)
        spectrum = spectrum[:, 14:]
        pointcloud = pointcloud[:, 14:]

        if self.calibration:
            spectrum = spectrum - self.calibration_spectrum
            pointcloud = torch.where(pointcloud == 1, torch.tensor(0), pointcloud)
            pointcloud = torch.where(pointcloud == 2, torch.tensor(1), pointcloud)

        # input preparation
        spectrum = spectrum.flatten()
        pointcloud = pointcloud.flatten()

        with torch.no_grad():
            spectrum_input = spectrum.unsqueeze(0)
            spectrum_input = spectrum_input.to(self.device)
            pointcloud_pred = self.model(spectrum_input)

        pointcloud_pred_np = pointcloud_pred.squeeze().detach().cpu().view(87, 128).numpy()
        pointcloud_gt_np = pointcloud.view(87, 128).numpy()
        spectrum_np = spectrum.view(87, 128).numpy()

        # Finding the two largest closures (bounding boxes) in the ground truth
        largest_bbox, second_largest_bbox = self.find_two_largest_closures(pointcloud_gt_np)

        # Extend the bounding boxes based on azimuth intervals
        if largest_bbox:
            largest_bbox = self.extend_bbox_to_azimuth_intervals(largest_bbox)
        if second_largest_bbox:
            # Extend the second largest azimuth using the largest_bbox's azimuth range
            r1, c1, r2, c2 = largest_bbox[0].start, largest_bbox[1].start, largest_bbox[0].stop, largest_bbox[1].stop
            second_largest_bbox = (slice(second_largest_bbox[0].start, second_largest_bbox[0].stop), slice(c1, c2))

        # Combine both bounding boxes into a list for easier handling
        bboxes = [largest_bbox, second_largest_bbox] if second_largest_bbox else [largest_bbox]

        # Visualization code (ensuring axes are always defined)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(spectrum_np, cmap='viridis', aspect='auto')
        axes[0].set_title('Spectrum')

        axes[1].imshow(pointcloud_pred_np, cmap='viridis', aspect='auto')
        axes[1].set_title('Predicted Pointcloud')

        axes[2].imshow(pointcloud_gt_np, cmap='gray', aspect='auto')
        axes[2].set_title('Ground Truth Pointcloud')

        # Draw bounding boxes for both the largest and second-largest closures
        colors = ['red', 'blue']  # Different colors for the bounding boxes
        for i, bbox in enumerate(bboxes):
            if bbox:
                r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop
                rect = patches.Rectangle((c1, r1), c2 - c1, r2 - r1, linewidth=2, edgecolor=colors[i], facecolor='none')
                axes[2].add_patch(rect)

        # Find the clustered maxima using DBSCAN
        max_positions = self.find_clustered_maxima(pointcloud_pred_np)

        # Log the found positions
        for i, max_position in enumerate(max_positions):
            logger.info(f'Clustered maximum {i + 1}: {max_position}')

        # Check if any of the clustered maxima are inside the bounding boxes
        inside_bbox = False
        for max_position in max_positions:
            for bbox in bboxes:
                if bbox:
                    r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop
                    if (r1 <= max_position[0] < r2) and (c1 <= max_position[1] < c2):
                        inside_bbox = True
                        break
            if inside_bbox:
                break  # Stop once one max_position is inside a bbox

        logger.info(f'Is any maximum value inside a bounding box? {"Yes" if inside_bbox else "No"}')

        # Accumulate the statistics for the full dataset
        self.total_gt_positives += np.sum(pointcloud_gt_np == 1)  # Ground truth positives
        if inside_bbox:
            self.total_true_positives += 1  # True positive if max_position is inside the bbox
        else:
            self.total_false_positives += 1  # False positive if max_position is outside the bbox

        self.total_predicted_positives += 1  # Count this prediction for false alarm calculation

        # Draw all detected maxima positions
        for max_position in max_positions:
            axes[2].plot(max_position[1], max_position[0], 'bo', markersize=8, label='Max Value')

        # Add a legend to indicate the maximum values
        axes[2].legend()

        # Save the figure to the given folder path with an automatic name
        save_folder = '/data/lucayu/lss-cfar/evaluation_visualization/lucacx_corridor_2024-08-27/'
        os.makedirs(save_folder, exist_ok=True)
        file_name = f'sample_{idx:04d}.png'
        save_path = os.path.join(save_folder, file_name)
        fig.savefig(save_path)

        logger.info(f'Visualization saved to {save_path}')

        plt.close(fig)  # Close the figure to free memory

        return 0

    def extend_bbox_to_azimuth_intervals(self, bbox):
        # Original bounding box coordinates
        r1, c1, r2, c2 = bbox[0].start, bbox[1].start, bbox[0].stop, bbox[1].stop

        # The resolution for azimuth is 15 bins, so we extend the bounding box accordingly.
        # Extend r1 and r2 to the nearest multiple of 15.
        r1_extended = max(0, r1 - (r1 % 10))
        r2_extended = min(86, r2 + (10 - (r2 % 10)))  # Ensure r2 stays within bounds

        # Return the extended bounding box
        return (slice(r1_extended, r2_extended), slice(c1, c2))

    def find_two_largest_closures(self, matrix):
        # Label connected components of 1s
        labeled_array, num_features = label(matrix)

        # Find bounding boxes of labeled components
        bounding_boxes = find_objects(labeled_array)

        # Count the number of value-1 pixels within each bounding box
        closures = []
        for bbox in bounding_boxes:
            if bbox is not None:
                closure_area = np.sum(matrix[bbox] == 1)
                closures.append((closure_area, bbox))

        # Sort the closures based on the number of value-1 pixels in descending order
        closures.sort(reverse=True, key=lambda x: x[0])

        # Ensure there are at least two closures
        if len(closures) >= 2:
            largest_bbox = closures[0][1]
            second_largest_bbox = closures[1][1]
            return largest_bbox, second_largest_bbox
        elif len(closures) == 1:
            return closures[0][1], None
        else:
            return None, None



    def find_clustered_maxima(self, pointcloud_pred_np, eps=10, min_samples=1):
        # Threshold to focus on high-intensity regions (you can adjust this threshold)
        threshold = np.percentile(pointcloud_pred_np, 99)  # Focus on the top 1% of the values
        high_value_points = np.column_stack(np.nonzero(pointcloud_pred_np >= threshold))

        if len(high_value_points) == 0:
            return []  # No significant points found

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(high_value_points)

        # Find centroids of each cluster
        clusters = {}
        for label in np.unique(clustering.labels_):
            if label == -1:  # Ignore noise points
                continue
            cluster_points = high_value_points[clustering.labels_ == label]
            centroid = np.mean(cluster_points, axis=0).astype(int)
            clusters[label] = centroid

        return list(clusters.values())  # Return list of centroids (maxima positions)

    def get_final_metrics(self):
        # Calculate detection rate and false alarm rate after processing the full dataset
        detection_rate = self.total_true_positives / self.__len__()
        false_alarm_rate = self.total_false_positives / self.__len__()

        logger.info(f"Final Detection Rate: {detection_rate:.4f}")
        logger.info(f"Final False Alarm Rate: {false_alarm_rate:.4f}")

        return detection_rate, false_alarm_rate
    
if __name__ == '__main__':
    dataset_path = '/data/lucayu/lss-cfar/dataset/lucacx_corridor_2024-08-27'
    checkpoint_path = '/home/lucayu/lss-cfar/checkpoints/20240828-010835_model_layers_4_hidden_256_order_256_dtmin_0.001_dtmax_8e-05_channels_1_dropout_0.1_lr_0.01_batch_4_steps_10000_optimizer_AdamW_decay_0.1_step_300_gamma_0.5_losstype_bce/20240828-010835_model_layers_4_hidden_256_order_256_dtmin_0.001_dtmax_8e-05_channels_1_dropout_0.1_lr_0.01_batch_4_steps_10000_optimizer_AdamW_decay_0.1_step_300_gamma_0.5_losstype_bce.pt'
    eval_dataset = evaluator(phase='test', dataset_path=dataset_path, checkpoint_path=checkpoint_path)

    # Create a DataLoader to handle batches (you can adjust batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Process all samples
    for data in tqdm(eval_loader):
        pass  # The __getitem__ method already handles the processing and accumulation

    # After processing all batches, get the final metrics
    detection_rate, false_alarm_rate = eval_dataset.get_final_metrics()