import numpy as np
from scipy import stats
from .types import NoiseModel


class Chi_1_2(NoiseModel):
    def pdf(self, x: np.ndarray, *, loc: float = 0, scale: float = 1.0) -> np.ndarray:
        return stats.chi2.pdf(x, df=1, loc=loc, scale=scale)

    def cdf(self, x: np.ndarray, *, loc: float = 0, scale: float = 1.0) -> np.ndarray:
        return stats.chi2.cdf(x, df=1, loc=loc, scale=scale)

    def ppf(
        self, alpha: np.ndarray, *, loc: float = 0, scale: float = 1.0
    ) -> np.ndarray:
        return stats.chi2.ppf(alpha, df=1, loc=loc, scale=scale)
