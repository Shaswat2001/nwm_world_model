import torch
import numpy as np

def extract_tensor_from_value(arr, timesteps, broadcast_shape):

    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]

    return res + torch.zeros(broadcast_shape, device=timesteps.device)

def mean_flat(tensor):

    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def gaussian_kl(mean1, logvar1, mean2, logvar2):

    def to_tensor(x, ref=None):
        if torch.is_tensor(x):
            return x
        if ref is not None and torch.is_tensor(ref):
            return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)
        return torch.as_tensor(x)

    mean1 = to_tensor(mean1)
    logvar1 = to_tensor(logvar1, mean1)
    mean2 = to_tensor(mean2, mean1)
    logvar2 = to_tensor(logvar2, mean1)

    return 0.5 * (
        logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2).pow(2) * torch.exp(-logvar2)
        - 1.0
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs