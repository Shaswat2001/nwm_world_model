import numpy as np

class BaseBetaSchedular:

    def __init__(self, betas: np.ndarray):
        
        self.betas = betas.astype(np.float32)
        self.forward_calculate()

    def forward_calculate(self):

        self.alphas = 1 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas, axis=0)
        self.alpha_cumprod_previous = np.append(1.0, self.alpha_cumprod[:-1])
        self.alpha_cumprod_next = np.append(self.alpha_cumprod[1:], 0.0)

        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1 - self.alpha_cumprod)
        self.log_one_minus_alpha_cumprod = np.log(1 - self.alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = np.sqrt(1.0 / self.alpha_cumprod)
        self.sqrt_recip1_alpha_cumprod = np.sqrt(1.0 / self.alpha_cumprod - 1.0)
        self.coefficient1 = self.betas * np.sqrt(self.alpha_cumprod_previous) / (1 - self.alpha_cumprod)
        self.coefficient2 = (1 - self.alpha_cumprod_previous) * np.sqrt(self.alphas) / (1 - self.alpha_cumprod)

        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_previous) / (1.0 - self.alpha_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])) if len(self.posterior_variance) > 1 else np.array([])

class ConstantBetaSchedular(BaseBetaSchedular):

    def __init__(self, beta_start: float = 0.001, beta_end: float = 0.001, diffusion_steps: int = 1000, **kwargs):
        
        betas = beta_end * np.ones((diffusion_steps,))
        super().__init__(betas)

class LinearBetaSchedular(BaseBetaSchedular):

    def __init__(self, beta_start: float = 0.001, beta_end: float = 0.001, diffusion_steps: int = 1000, **kwargs):
        
        betas = np.linspace(start= beta_start, stop= beta_end, num= diffusion_steps, dtype=np.float32)
        super().__init__(betas)

class WarmupBetaSchedular(BaseBetaSchedular):

    def __init__(self, beta_start: float = 0.001, beta_end: float = 0.001, diffusion_steps: int = 1000, warmup_ratio: float = 0.1, **kwargs):
        
        assert warmup_ratio is not None

        betas = beta_end * np.ones((diffusion_steps,))
        warmup_steps = int(warmup_ratio * diffusion_steps)
        betas[:warmup_steps] = np.linspace(start= beta_start, stop= beta_end, num= warmup_steps, dtype=np.float32)
        super().__init__(betas)


schedulars = {
    "constant": ConstantBetaSchedular,
    "linear": LinearBetaSchedular,
    "warmup": WarmupBetaSchedular
}