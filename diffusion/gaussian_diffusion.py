import yaml
from pathlib import Path

import torch
import numpy as np
from diffusion.schedular import schedulars
from diffusion.diffusion_utils import extract_tensor_from_value, gaussian_kl, mean_flat, discretized_gaussian_log_likelihood

ROOT = Path(__file__).resolve().parent.parent

class GaussianDiffusion:

    def __init__(self):
        config_path = ROOT / "config" / "diffusion_cfg.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.schedular = schedulars[self.config["schedular"]["type"]](**self.config["schedular"])
        self.diffusion_steps = self.config["schedular"]["diffusion_steps"]

    def q_mean_variance(self, x_start, t):

        mean = extract_tensor_from_value(self.schedular.sqrt_alpha_cumprod, t, x_start.shape) * x_start
        variance = extract_tensor_from_value(self.schedular.sqrt_one_minus_alpha_cumprod, t, x_start.shape)
        log_variance = extract_tensor_from_value(self.schedular.log_one_minus_alpha_cumprod, t, x_start.shape)

        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise = None):

        if noise is None:
            noise = torch.randn_like(x_start, dtype=torch.float32)
        
        assert noise.shape == x_start.shape

        return extract_tensor_from_value(self.schedular.sqrt_alpha_cumprod, t, x_start.shape) * x_start + \
                extract_tensor_from_value(self.schedular.sqrt_one_minus_alpha_cumprod, t, x_start.shape) * noise
    
    def q_posterior_mean_variance(self, x_t, x_start, t):

        assert x_t.shape == x_start.shape

        mean = extract_tensor_from_value(self.schedular.coefficient1, t, x_start.shape) * x_start + \
            extract_tensor_from_value(self.schedular.coefficient2, t, x_start.shape) * x_t
        
        variance = extract_tensor_from_value(self.schedular.posterior_variance, t, x_start.shape)
        log_variance = extract_tensor_from_value(self.schedular.posterior_log_variance_clipped, t, x_start.shape)

        return mean, variance, log_variance
    
    def calculate_x_start_from_epsilon(self, x_t, t, epsilon):
        assert x_t.shape == epsilon.shape
        return extract_tensor_from_value(self.schedular.sqrt_recip_alpha_cumprod, t, x_t.shape) * x_t - \
            extract_tensor_from_value(self.schedular.sqrt_recip1_alpha_cumprod, t, x_t.shape) * epsilon
    
    def calculate_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_tensor_from_value(self.sqrt_recip_alpha_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract_tensor_from_value(self.sqrt_recip1_alpha_cumprod, t, x_t.shape)
    
    def process_state(self, x, denoised_clip: bool = False, denoise_fun = None):

        if denoise_fun is not None:
            x = denoise_fun(x)

        if denoised_clip:
            return x.clamp(-1, 1)
        
        return x

    def p_mean_variance(self, model, x, t, denoised_clip: bool = False, denoise_fun = None, model_kwargs=None):
        
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        model_output = model(x, t, **model_kwargs)

        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.config["sigma"] == "learned" or self.config["sigma"] == "learned_scaled":
            assert model_output.shape == (B, 2*C, *x.shape[2:])
            model_output, model_variance = torch.split(model_output, C, dim= 1)

            log1 = extract_tensor_from_value(np.log(self.schedular.betas), t, x.shape)
            log2 = extract_tensor_from_value(self.schedular.posterior_log_variance_clipped, t, x.shape)

            fraction = (model_variance + 1) / 2.0
            log_variance = fraction * log1 + (1 - fraction) * log2
            variance = torch.exp(log_variance)
        else:

            if self.config["sigma"] == "fixed_small":

                log_variance = self.schedular.posterior_log_variance_clipped
                variance = self.schedular.posterior_variance
            else:
                variance = np.append(self.schedular.posterior_variance[1], self.schedular.betas[1:]),
                log_variance = np.log(np.append(self.schedular.posterior_variance[1], self.schedular.betas[1:]))
            
            variance = extract_tensor_from_value(variance, t, x.shape)
            log_variance = extract_tensor_from_value(log_variance, t, x.shape)
        
        if self.config["model_output"] == "start_x":
            pred_x_start = self.process_state(model_output, denoised_clip, denoise_fun)
        else:
            pred_x_start = self.process_state(self.calculate_x_start_from_epsilon(x, t, epsilon=model_output), denoised_clip, denoise_fun)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x, pred_x_start, t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": log_variance,
            "pred_xstart": pred_x_start,
            "extra": extra,
        }
    
    def p_sample(self, model, x, t, denoised_clip: bool = False, denoise_fun = None, model_kwargs=None):

        output = self.p_mean_variance(model, x, t, denoised_clip, denoise_fun, model_kwargs)
        noise = torch.randn_like(x, dtype=torch.float32)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        sample = output["mean"] + nonzero_mask * torch.exp(0.5 * output["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": output["pred_xstart"]}
    
    def p_reverse(self, model, shape, noise = None, denoised_clip: bool = False, denoise_fun = None, model_kwargs=None):

        assert isinstance(shape, (tuple, list))

        timesteps = list(reversed(range(self.diffusion_steps)))

        if noise is None:
            img = torch.randn_like(*shape, dtype=torch.float32)
        else:
            img = noise

        for t in timesteps:

            output = self.p_sample(model, img, t, denoised_clip, denoise_fun, model_kwargs)
            img = output["sample"]

        return img
    
    def variational_lower_bound(self, model, x_start, x_t, t, denoised_clip=True, model_kwargs=None):

        true_mean, true_variance, true_log_variance = self.q_posterior_mean_variance(x_t, x_start, t)

        output = self.p_mean_variance(model, x_t, t, denoised_clip, None, model_kwargs)

        kl = gaussian_kl(
            true_mean, true_log_variance, output["mean"], output["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=output["mean"], log_scales=0.5 * output["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        loss = torch.where((t == 0), decoder_nll, kl)
        return {"output": loss, "pred_xstart": output["pred_xstart"]}

    def diffusion_loss(self, model, x_start, t, model_kwargs= None, noise= None):

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.config["loss_type"] == "kl" or self.config["loss_type"] == "rescaled_kl":
            terms["loss"] = self.variational_lower_bound(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                denoised_clip=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.config["loss_type"] == "rescaled_kl":
                terms["loss"] *= self.diffusion_steps

        else:
            raise NotImplementedError(self.config["loss_type"])

        return terms

        
        

        