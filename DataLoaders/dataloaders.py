import os
from bisect import bisect_left
## Standard libraries
import cv2
import numpy as np
import os, random, tarfile, gc, timm #timm to install DinoV2
from IPython.display import display, clear_output
from PIL import Image
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader


def extract_timestamp(path):
    return int(os.path.splitext(os.path.basename(path))[0])

def match_lidar_to_closest_images(lidar_paths, image_paths):
    """
    Match each LiDAR timestamp to the closest image.
    Returns a list of matched image paths in same order as lidar_paths.
    """
    image_ts_paths = sorted([(extract_timestamp(p), p) for p in image_paths])
    image_ts = [ts for ts, _ in image_ts_paths]
    image_paths_sorted = [p for _, p in image_ts_paths]

    matched_images = []

    for lidar_path in lidar_paths:
        ts = extract_timestamp(lidar_path)
        idx = bisect_left(image_ts, ts)
        if idx == 0:
            best_idx = 0
        elif idx == len(image_ts):
            best_idx = len(image_ts) - 1
        else:
            before = image_ts[idx - 1]
            after = image_ts[idx]
            best_idx = idx if abs(after - ts) < abs(before - ts) else idx - 1

        matched_images.append(image_paths_sorted[best_idx])

    return matched_images

class LiDARDataset(Dataset):
    def __init__(self, lidar_paths):
        self.lidar_paths = lidar_paths

    def __len__(self):
        return len(self.lidar_paths)

    def __getitem__(self, idx):
        lidar_df = feather.read_feather(self.lidar_paths[idx])
        lidar_tensor = torch.tensor(lidar_df[['x', 'y', 'z']].to_numpy(), dtype=torch.float32)
        return lidar_tensor


class ImageDataset(Dataset):
    def __init__(self, camera_paths_dict, image_size):
        self.camera_paths_dict = camera_paths_dict
        self.camera_order = [
            'ring_front_left', 'ring_front_center', 'ring_front_right',
            'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right'
        ]
        self.length = len(next(iter(camera_paths_dict.values())))
        self.resize = transforms.Resize((image_size, image_size))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        imgs = []
        for cam in self.camera_order:
            img_path = self.camera_paths_dict[cam][idx]
            img = mpimg.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img, dtype=torch.float32) / 255.0
            img = self.resize(img)
            imgs.append(img)
        return torch.stack(imgs, dim=0)  # [7, 3, H, W]

class MultiGroupDataset(Dataset):
    def __init__(self, lidar_dataset:LiDARDataset=None, image_dataset:ImageDataset=None, mode='both'):
        assert mode in ['lidar', 'image', 'both']
        self.mode = mode
        self.lidar_dataset = lidar_dataset
        self.image_dataset = image_dataset
        self.length = len(lidar_dataset or image_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'lidar':
            return self.lidar_dataset[idx]
        elif self.mode == 'image':
            return self.image_dataset[idx]
        elif self.mode == 'both':
            return self.lidar_dataset[idx], self.image_dataset[idx]

