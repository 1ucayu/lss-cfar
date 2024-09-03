import torch
import numpy as np
from loguru import logger
import os
import io
import pickle
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class CNNDataset(Dataset):
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
                self.calibration_spectrums.append(calibration_spectrum)  # Correctly place this line inside the loop

    def __len__(self):
        logger.info(f'Dsataset Loaded, Size: {len(self.data_list)}')
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_path = self.data_list[idx]

        # Determine which dataset and corresponding calibration spectrum to use
        if self.calibration:
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

            # for pointcloud, if calibration, set the 1 value to 0 then set 2 value to 1
        pointcloud = torch.where(pointcloud == 1, torch.tensor(0), pointcloud)
        pointcloud = torch.where(pointcloud == 2, torch.tensor(1), pointcloud)

        # Flatten the spectrum and pointcloud
        # spectrum = spectrum.flatten()
        # pointcloud = pointcloud.flatten()
        
        pointcloud = pointcloud.unsqueeze(0)
        spectrum = spectrum.unsqueeze(0)
        # normalize the spectrum
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())

        logger.info(f'pointcloud shape: {pointcloud.shape}')
        logger.info(f'spectrum shape: {spectrum.shape}')
        
        result = {
            'pointcloud': pointcloud,
            'spectrum': spectrum
        }
        return result
    
    def _collate_fn(self, batch):
        # Stack pointclouds along the batch dimension (second dimension)
        pointclouds = torch.stack([item['pointcloud'] for item in batch], dim=0)
        spectrums = torch.stack([item['spectrum'] for item in batch], dim=0)
        return {'pointcloud': pointclouds, 'spectrum': spectrums}


# Test function
def test():
    dataset_path = ['/data/lucayu/lss-cfar/dataset/lucacx_corridor_2024-08-27']
    calibration_paths = ['/data/lucayu/lss-cfar/raw_dataset/lucacx_env_corridor_2024-08-27']
    phase = 'train'
    dataset = CNNDataset(phase, dataset_path, calibration_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset._collate_fn)  # Adjust batch size as needed

    logger.info(f'len(dataset): {len(dataset)}')
    for batch in tqdm(dataloader):
        logger.info(f'Batch pointcloud shape: {batch["pointcloud"].shape}')  # Example shape: (1, batch_size, H, W)
        logger.info(f'Batch spectrum shape: {batch["spectrum"].shape}')  # Example shape: (1, batch_size, C, H, W)
        break


if __name__ == '__main__':
    test()
