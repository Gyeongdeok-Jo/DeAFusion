from torch.utils.data import Dataset
from glob import glob
import os
from utils.util import *
import cv2
import numpy as np
from collections import OrderedDict

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, modality, transforms = None, grayscale = False):
        self.modality_paths = {}
        for m in modality:
            self.modality_paths[m] = sorted(glob(os.path.join(data_dir, f'*{m}*.png')))
        self.length = len(self.modality_paths[m])
        self.transforms = transforms
        self.mask_keys = [key for key, value in transforms.additional_targets.items() if value == 'mask']
        self.grayscale = grayscale

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = OrderedDict()
        for key, value in self.modality_paths.items():
            if key in self.mask_keys:
                image = cv2.imread(value[idx], cv2.IMREAD_GRAYSCALE)[:,:, np.newaxis]
                image[image>0] = 1
            else:
                if self.grayscale:
                    image = min_max_scaling(cv2.imread(value[idx], cv2.IMREAD_GRAYSCALE))
                else:
                    image = min_max_scaling(cv2.imread(value[idx]))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data[key] = image
        agumented_data = self.transforms(image = np.zeros_like(data[key]), **data)
    
        del agumented_data['image']
        del data
        return agumented_data, value[idx]
