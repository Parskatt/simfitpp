import numpy as np

from .noise_models import Chi_1_2
from .types import NoiseModel, ScaleEstimator


class UnbiasedMed(ScaleEstimator):
    def __call__(
        self, noise_model: NoiseModel, residuals_squared: np.ndarray, th_squared: float
    ) -> float:
        if residuals_squared is None:
            return np.nan
        if isinstance(noise_model, Chi_1_2):
            # below 4 there is no fix point for chi_1, 5 seems like a reasonable place to give up.
            # TODO: not clear we need to do this, est will likely be filtered out anyway...
            min_ratio = 5 
        else:
            min_ratio = 0
        residuals_squared = residuals_squared[residuals_squared < th_squared]
        r_sqr_med = np.median(residuals_squared)
        ratio = th_squared / r_sqr_med
        q = 0.5
        sigma_sqr0 = r_sqr_med / noise_model.ppf(q)
        if ratio < min_ratio:
            return sigma_sqr0
        sigma_sqr = sigma_sqr0
        for _ in range(5):
            q = noise_model.cdf(ratio * noise_model.ppf(q)) / 2
        sigma_sqr = r_sqr_med / noise_model.ppf(q)
        return sigma_sqr
