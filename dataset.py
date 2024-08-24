import torch
import numpy as np
from loguru import logger
import os
import io
import pickle
from PIL import Image
import tqdm
from torch.utils.data import DataLoader, Dataset


class lsscfarDataset(Dataset):
    def __init__(self, phase, dataset_path):
        self.phases = phase
        self.data_list = []
        self.dataset_path = dataset_path
        data_path = f'{dataset_path}/{phase}'
        self.data_list += [f'{dataset_path}/{phase}/{data}' for data in os.listdir(data_path)]
        self.calibration = True

    def __len__(self):
        logger.info(f'Dataset Loaded, Size: {len(self.data_list)}')
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        # raw_pointcloud = data['pointcloud']
        spectrum = torch.tensor(data['spectrum'], dtype=torch.float32)
        pointcloud = torch.tensor(data['pointcloud'], dtype=torch.float32)
        # normalize the spectrum
        if self.calibration:
            # load the calibration data: /data/lucayu/lss-cfar/raw_dataset/luca_env_hw_101_2024-08-23_21-39-12.pickle
            calibration_path = '/data/lucayu/lss-cfar/raw_dataset/luca_env_hw_101_2024-08-23_21-39-24.pickle'
            with open(calibration_path, 'rb') as f:
                calibration_data = pickle.load(f)
            calibration_spectrum = torch.tensor(calibration_data['spectrum'], dtype=torch.float32)
            # flip the calibration spectrum y-axis
            calibration_spectrum = calibration_spectrum.flip(0)
            # calibrate the spectrum by subtracting the calibration spectrum
            spectrum = spectrum - calibration_spectrum

            # for pointcloud, if calibration, set the 1 value to 0 then set 2 value to 1
            pointcloud = torch.where(pointcloud == 1, torch.tensor(0), pointcloud)
            pointcloud = torch.where(pointcloud == 2, torch.tensor(1), pointcloud)
        
        # normalize the spectrum
        ############## do not normalize the spectrum
        # spectrum = spectrum / spectrum.max()
        # z-score normalization
        # spectrum = (spectrum - spectrum.mean()) / spectrum.std()
        # min-max normalization
        # spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())

        # pointcloud = self._pointcloud_process(raw_pointcloud, spectrum)
        # logger.info(f'Pointcloud shape: {pointcloud.shape}')
        
        # flatten the spectrum and pointcloud
        spectrum = spectrum.flatten()
        pointcloud = pointcloud.flatten()
        # logger.info(f'Pointcloud shape: {pointcloud.shape}')
        result = {
            'pointcloud': pointcloud,    #.transpose(0, -1),
            'spectrum': spectrum    #.transpose(0, -1)
        }
        return result
    
    def _pointcloud_process(self, pointcloud, spectrum):
        pointcloud = Image.open(io.BytesIO(pointcloud))
        pointcloud = pointcloud.resize((spectrum.shape[1], spectrum.shape[0]))
        pointcloud = torch.from_numpy(np.array(pointcloud)).float()
        pointcloud = (pointcloud[:, :, 3] > 0).float()  # > / == debug here
        return pointcloud

    def _collate_fn(self, batch):
        # Stack pointclouds along the batch dimension (second dimension)
        pointclouds = torch.stack([item['pointcloud'] for item in batch], dim=1)  # Batch dimension is 1
        spectrums = torch.stack([item['spectrum'] for item in batch], dim=1)  # Batch dimension is 1
        return {'pointcloud': pointclouds, 'spectrum': spectrums}


# Test function
def test():
    dataset_path = '/data/lucayu/lss-cfar'
    phase = 'train'
    dataset = lsscfarDataset(phase, dataset_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset._collate_fn)  # Adjust batch size as needed

    logger.info(f'len(dataset): {len(dataset)}')
    for batch in tqdm.tqdm(dataloader):
        logger.info(f'Batch pointcloud shape: {batch["pointcloud"].shape}')  # Example shape: (1, batch_size, H, W)
        logger.info(f'Batch spectrum shape: {batch["spectrum"].shape}')  # Example shape: (1, batch_size, C, H, W)
        break


if __name__ == '__main__':
    test()
