from abc import ABC, abstractmethod
import numpy as np


class NoiseModel(ABC):
    @abstractmethod
    def pdf(self, x: float, *, loc: float = 0, scale: float = 1.0) -> float:
        pass

    @abstractmethod
    def cdf(self, x: float, *, loc: float = 0, scale: float = 1.0) -> float:
        pass

    @abstractmethod
    def ppf(self, alpha: float, *, loc: float = 0, scale: float = 1.0) -> float:
        pass


class GeometricModel(ABC):
    @abstractmethod
    def compute_squared_residuals(
        self, data_dict: dict[str, np.ndarray], *, subset: np.ndarray = None
    ) -> np.ndarray:
        pass


class GeomEstimator(ABC):
    @abstractmethod
    def __call__(
        self, data_dict: dict[str, np.ndarray], *args, **kwds
    ) -> GeometricModel:
        pass

    def load_corresps(
        self, data_dict: dict[str, np.ndarray], *, subset: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        x_A, x_B = data_dict["x1"][:], data_dict["x2"][:]
        if subset is not None:
            x_A, x_B = x_A[subset], x_B[subset]
        return x_A, x_B


class ScaleEstimator(ABC):
    @abstractmethod
    def __call__(
        self, noise_model: NoiseModel, residuals_squared: np.ndarray, th_squared: float
    ) -> float:
        pass
