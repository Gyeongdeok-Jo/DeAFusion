import numpy as np
import yaml


def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val + 1e-7)
    return scaled_data

def create_dict(keys):
    result = {}
    for key in keys:
        if key in ['T1', 'FLAIR']:
            result[key] = 'image'
        else:
            result[key] = 'mask'
    return result
   

def overlay_plot(image, label, color):       # red or yellow
    # label = cv2.cvtColor(label.astype(np.float32), cv2.COLOR_GRAY2RGB)
    image = (image*255.).astype(np.uint8)
    image[label.squeeze(2)>0] = color

    return image


class build_total_loss_dict:
    def __init__(self, keys):
        self.total_loss_dict = {}
        for key in keys:
            self.total_loss_dict[key] = []
        self.keys = self.total_loss_dict.keys()
    def initial_dict(self):
        for key in self.keys:
            self.total_loss_dict[key] = []

    def append_value(self, log_dict):
        for key in self.keys:
            self.total_loss_dict[key].append(log_dict[key])
    def intergrate(self, cnt):
        for key in self.keys:
            self.total_loss_dict[key] = sum(self.total_loss_dict[key])/cnt


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(exc)
