from torch.utils.data import Dataset, DataLoader
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
from loguru import logger
from tqdm import tqdm

class dataset_processing(Dataset):
    def __init__(self, raw_dataset_path, save_processed_folder_path, save_processed_save_visualization_folder):
        self.data_list = []
        self.save_processed_folder_path = save_processed_folder_path
        self.save_processed_save_visualization_folder = save_processed_save_visualization_folder

        # Ensure the directories exist, create them if not
        os.makedirs(self.save_processed_folder_path, exist_ok=True)
        os.makedirs(self.save_processed_save_visualization_folder, exist_ok=True)

        # Read all *.pickle absolute file paths under raw_dataset_path
        for root, dirs, files in os.walk(raw_dataset_path):
            for file in files:
                if file.endswith('.pickle'):
                    self.data_list.append(os.path.join(root, file))
        logger.info(f"Total {len(self.data_list)} files found in {raw_dataset_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_path = self.data_list[idx]
        spectrum, raw_pointcloud, depth_image, color_image, combination = self.load_pickle(sample_path)
        self.visualize_data(sample_path, spectrum, raw_pointcloud, depth_image, color_image, combination, self.save_processed_folder_path, self.save_processed_save_visualization_folder)
        return 0

    def load_pickle(self, file_path):
        """Load the pickle file from the given path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        spectrum = data.get('spectrum')
        raw_pointcloud = data.get('raw_pointcloud')
        depth_image = data.get('depth_image')
        color_image = data.get('color_image')
        combination = data.get('combination')

        return spectrum, raw_pointcloud, depth_image, color_image, combination

    def cartesian_to_polar(self, x, y, z):
        """Convert Cartesian coordinates to polar coordinates for point cloud."""
        range_ = np.sqrt(x**2 + z**2)
        azimuth = np.arctan2(-x, z)  # Azimuth remains in radians for polar plot

        mask = (range_ >= 0) & (range_ <= 6)  # Adjust based on your setup's max range
        range_, azimuth = range_[mask], azimuth[mask]

        azimuth_deg = np.degrees(azimuth)
        return range_, azimuth, azimuth_deg

    def calculate_point_cloud(self, depth_frame, intrinsics, depth_scale):
        height, width = depth_frame.shape
        fx, fy = intrinsics['fx'], intrinsics['fy']
        ppx, ppy = intrinsics['ppx'], intrinsics['ppy']
        x_indices = np.tile(np.arange(width), height).reshape(height, width)
        y_indices = np.repeat(np.arange(height), width).reshape(height, width)
        z = depth_frame * depth_scale
        x = (x_indices - ppx) * z / fx
        y = (y_indices - ppy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        return points

    def visualize_data(self, file_path, spectrum, raw_pointcloud, depth_image, color_image, combination, save_processed_folder_path, save_processed_save_visualization_folder):
        fig = plt.figure(figsize=(60, 30), constrained_layout=True)

        ax1 = fig.add_subplot(2, 6, 1)
        ax2 = fig.add_subplot(2, 6, 2, polar=True)
        ax3 = fig.add_subplot(2, 6, 3)
        ax4 = fig.add_subplot(2, 6, 4)
        ax5 = fig.add_subplot(2, 6, 5)
        ax6 = fig.add_subplot(2, 6, 6)
        ax7 = fig.add_subplot(2, 6, 7)
        ax8 = fig.add_subplot(2, 6, 8)
        ax9 = fig.add_subplot(2, 6, 9, polar=True)
        ax10 = fig.add_subplot(2, 6, 10)
        ax11 = fig.add_subplot(2, 6, 11)
        ax12 = fig.add_subplot(2, 6, 12)

        # ax1: Radar Spectrum
        ax1.imshow(spectrum, aspect='auto', cmap='viridis')

        # ax2: Raw Point Cloud (Polar)
        x, y, z = raw_pointcloud[:, 0], raw_pointcloud[:, 1], raw_pointcloud[:, 2]
        range_, azimuth, _ = self.cartesian_to_polar(x, y, z)
        
        ax2.scatter(azimuth, range_, s=1, c='red')
        ax2.set_ylim(0, 6)  # Adjust range limits based on your setup
        ax2.set_xlim(-87 / 2 * np.pi / 180, 87 / 2 * np.pi / 180)

        # ax3: Depth Image
        flipped_depth_image = cv2.flip(depth_image, 0)
        ax3.imshow(flipped_depth_image, cmap='gray')

        # ax4: Color Image
        flipped_color_image = cv2.flip(color_image, 0)
        ax4.imshow(cv2.cvtColor(flipped_color_image, cv2.COLOR_BGR2RGB))

        # ax5: Combined Image
        combination_img = plt.imread(io.BytesIO(combination), format='png')
        ax5.imshow(combination_img)

        # Load a pre-recorded depth image
        depth_image = depth_image.astype(np.float32)  # Convert to float32 for scaling

        # Manually set the depth camera intrinsics and scale
        depth_intrinsics = {
            'width': 640,
            'height': 480,
            'fx': 387.7923278808594,
            'fy': 387.7923278808594,
            'ppx': 322.8212890625,
            'ppy': 240.0816650390625
        }
        depth_scale = 0.0010000000474974513

        # Calculate the 3D point cloud from the depth image
        verts = self.calculate_point_cloud(depth_image, depth_intrinsics, depth_scale)

        spectrum_shape = spectrum.shape
        result = np.zeros(spectrum_shape)

        # Scale the azimuth and range to match the spectrum resolution
        azimuth_indices = np.clip(((azimuth + 87/2) / 87 * spectrum_shape[0]).astype(int), 0, spectrum_shape[0] - 1)
        range_indices = np.clip((range_ / 6 * spectrum_shape[1]).astype(int), 0, spectrum_shape[1] - 1)

        # Assign values to the point cloud in the result matrix
        result[azimuth_indices, range_indices] = 1  # Background points valued as 1

        # ax11: Person and Background Point Cloud
        ax11.imshow(result, cmap='gray', origin='lower')

        # ax12: Spectrum Plot (with Overlaid Point Cloud)
        spectrum = np.flipud(spectrum)
        ax12.imshow(spectrum, aspect='auto', cmap='viridis', alpha=0.5)
        ax12.imshow(result, origin='lower', alpha=0.5)

        plt.tight_layout()
        plt.show()

        save_processed_path = save_processed_save_visualization_folder + '/' + os.path.basename(file_path).replace('.pickle', '.png')
        plt.savefig(save_processed_path)

        result_processed = {
            'spectrum': spectrum,
            'pointcloud': result
        }

        save_processed_path = save_processed_folder_path + '/' + os.path.basename(file_path)
        with open(save_processed_path, 'wb') as f:
            pickle.dump(result_processed, f)

        plt.close(fig)

if __name__ == "__main__":
    raw_dataset_path = '/data/lucayu/lss-cfar/raw_dataset/wayne_env_office_2024-08-27'
    save_processed_folder_path = '/data/lucayu/lss-cfar/dataset/wayne_env_office_2024-08-27'
    save_processed_save_visualization_folder = '/data/lucayu/lss-cfar/dataset_visualization/wayne_env_office_2024-08-27'

    # Initialize the dataset and dataloader
    dataset = dataset_processing(raw_dataset_path, save_processed_folder_path, save_processed_save_visualization_folder)
    dataloader = DataLoader(dataset, batch_size=12, num_workers=0, shuffle=False)

    # Use tqdm to show the progress
    for i, data in enumerate(tqdm(dataloader)):
        pass
