from .scale_estimators import UnbiasedMed
from .types import GeomEstimator
import numpy as np


class SIMFITPP:
    def __init__(
        self,
        geom_estimator: GeomEstimator,
        *,
        alpha: float = 0.99,
        max_iter: int = 4,
        ftol: float = 0.01,
        train_fraction: float = 0.5,
        th_min: float = 0.25,
        th_max: float = 8,
    ):
        self.geom_estimator = geom_estimator
        self.scale_estimator = UnbiasedMed()
        self.alpha = alpha
        self.max_iter = max_iter
        self.ftol = ftol
        self.train_fraction = train_fraction
        self.th_min = th_min
        self.th_max = th_max

    def estimate_threshold(
        self,
        x_A: np.ndarray,
        x_B: np.ndarray,
        th_guess: float,
        num_previous_est: int = 0,
        *,
        K_A=None,
        K_B=None,
    ) -> tuple[float, bool]:
        """Estimates RANSAC Threshold from Correspondnces.

        Args:
            x_A (np.ndarray): Points in A.
            x_B (np.ndarray): Corresponding points in B.
            th_guess (float): An initial guess for the threshold.
            num_previous_est (int, optional): How many pairs the initial guess is based on. Defaults to 0.
            K_A (np.ndarray, optional): Intrinsics for A. Defaults to None.
            K_B (np.ndarray, optional): Intrinsics for B. Defaults to None.

        Returns:
            tuple[float, bool]: estimated threshold (float) and whether the estimate was successful (bool).
        """
        data_dict = {}
        data_dict["x1"] = x_A
        data_dict["x2"] = x_B
        data_dict["K1"] = K_A
        data_dict["K2"] = K_B
        return self(data_dict, th_guess, num_previous_est)

    def __call__(
        self,
        data_dict: dict[str, np.ndarray],
        th_guess: float,
        num_previous_est: int = 0,
    ) -> tuple[float, bool]:
        inner_num_success = 0
        th_old = th_guess
        seen_th = set()
        noise_model = self.geom_estimator.noise_model
        th_sqr_over_sigma_sqr = noise_model.ppf(self.alpha)
        th = np.nan
        corresps = np.concatenate((data_dict["x1"][:], data_dict["x2"][:]), axis=-1)
        for _ in range(self.max_iter):
            perm = np.random.permutation(len(corresps))
            train_end_ind = int(len(corresps) * self.train_fraction)
            train_set = perm[:train_end_ind]
            val_set = perm[train_end_ind:]
            # estimate geometry and compute residuals
            model = self.geom_estimator(data_dict, th_old, subset=train_set)
            residuals_squared_val = model.compute_squared_residuals(
                data_dict, subset=val_set
            )
            # estimate scale from estimated geometry and squared residuals + threshold
            sigma_sqr = self.scale_estimator(
                noise_model, residuals_squared_val, th_old**2
            )
            th_new = np.sqrt(th_sqr_over_sigma_sqr * sigma_sqr)
            in_bounds = th_new >= self.th_min and th_new <= self.th_max
            if not in_bounds:
                continue
            th = (
                1 / (inner_num_success + 1) * th_new
                + inner_num_success / (inner_num_success + 1) * th_old
            )
            if abs(th - th_old) < self.ftol:
                break
            elif f"{th:.3f}" in seen_th:
                break
            else:
                seen_th.add(f"{th:.3f}")
            inner_num_success += 1
            th_old = th
        final_th = th
        out_of_bounds: bool = (
            np.isnan(final_th) or final_th <= self.th_min or final_th >= self.th_max
        )
        is_success = not out_of_bounds
        if not is_success:
            final_th = th_guess
        else:
            # in multi-pair setting we filter with previous estimates
            final_th = np.exp(
                1 / (num_previous_est + 1) * np.log(final_th)
                + num_previous_est / (num_previous_est + 1) * np.log(th_guess)
            )
        return final_th, is_success
