import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import torchvision.transforms.functional as F
IMAGE_ASPECT_RATIO = (4 / 3)  # all images are centered cropped to a 4:3 aspect ratio in training

def get_image_path(data_folder, traj_name, t):
    return os.path.join(data_folder, traj_name, f"{t}.jpg")

class CenterCropAR:
    def __init__(self, ar: float = IMAGE_ASPECT_RATIO):
        self.ar = ar

    def __call__(self, img: Image.Image):
        w, h = img.size
        if w > h:
            img = F.center_crop(img, (h, int(h * self.ar)))
        else:
            img = F.center_crop(img, (int(w / self.ar), w))
        return img

transform = transforms.Compose([
    CenterCropAR(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def angle_difference(theta1, theta2):
    delta_theta = theta2 - theta1    
    delta_theta = delta_theta - 2 * np.pi * np.floor((delta_theta + np.pi) / (2 * np.pi))    
    return delta_theta

def get_delta_np(actions):
    # append zeros to first action (unbatched)
    ex_actions = np.concatenate((np.zeros((1, actions.shape[1])), actions), axis=0)
    delta = ex_actions[1:] - ex_actions[:-1]
    
    return delta

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

@torch.no_grad()
def update_target_networks(target_model, model, tau:float = 0.9999):

    target_params = OrderedDict(target_model.named_parameters())
    params = OrderedDict(model.named_parameters())

    for name, param in params.items():
        target_params[name].mul_(tau).add_(param.data, alpha=1 - tau)

def requires_grad(model, flag=True):

    for p in model.parameters():
        p.requires_grad = flag