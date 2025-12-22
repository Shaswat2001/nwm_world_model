import torch

def extract_tensor_from_value(arr, timesteps, broadcast_shape):

    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]

    return res + torch.zeros(broadcast_shape, device=timesteps.device)