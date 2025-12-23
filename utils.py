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

unnormalize = transforms.Normalize(
    mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)

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

def save_image(output_file, img, unnormalize_img):
    
    img = img.detach().cpu()
    if unnormalize_img:
        img = unnormalize(img)
        
    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

    image.save(output_file)

def save_image_rollouts(save_dir, indexs, sec, images):

    for batch_idx, sample_idx in enumerate(indexs.squeeze()):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(save_dir, f'id_{sample_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f'{sec}.png')
        save_image(image_file, images[batch_idx], True)

def nwm_model_forward(all_models, curr_obs, curr_delta, num_timesteps, latent_size, device, num_cond, num_goals=1, rel_t=None):
    
    model, tokenizer, diffusion = all_models
    x = curr_obs.to(device)
    y = curr_delta.to(device)

    B, T = x.shape[:2]

    if rel_t is None:
        rel_t = (torch.ones(B)* (1. / 128.)).to(device)
        rel_t *= num_timesteps

    x = x.flatten(0, 1)
    x = tokenizer.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
    x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
    z = torch.randn(B*num_goals, 4, latent_size, latent_size, device=device)
    y = y.flatten(0, 1)
    model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)      
    samples = diffusion.p_reverse(model.forward, z.shape, z, denoised_clip=False, model_kwargs=model_kwargs)
    samples = tokenizer.decode(samples / 0.18215).sample

    return torch.clip(samples, -1., 1.)


