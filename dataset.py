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

    def __len__(self):
        # logger.info(f'Dataset Loaded, Size: {len(self.data_list)}')
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        raw_pointcloud = data['pointcloud']
        spectrum = torch.tensor(data['spectrum'], dtype=torch.float32)
        pointcloud = self._pointcloud_process(raw_pointcloud, spectrum)
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
    phase = ['train', 'test']
    dataset = lsscfarDataset(phase, dataset_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset._collate_fn)  # Adjust batch size as needed

    logger.info(f'len(dataset): {len(dataset)}')
    for batch in tqdm.tqdm(dataloader):
        logger.info(f'Batch pointcloud shape: {batch["pointcloud"].shape}')  # Example shape: (1, batch_size, H, W)
        logger.info(f'Batch spectrum shape: {batch["spectrum"].shape}')  # Example shape: (1, batch_size, C, H, W)
        break


if __name__ == '__main__':
    test()
