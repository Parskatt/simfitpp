import cv2
import numpy as np
from .types import GeometricModel
from .utils import skew_matrix, to_homog, calibrate_points

class Essential(GeometricModel):
    def __init__(
        self,
        *,
        K_A: np.ndarray = None,
        K_B: np.ndarray = None,
        rotation: np.ndarray = None,
        translation: np.ndarray = None,
    ):
        super().__init__()
        self.K_A = K_A  # 3 x 3
        self.K_B = K_B  # 3 x 3
        self.rotation = rotation  # 3 x 3
        self.translation = translation  # 3

    @classmethod
    def from_fundamental_and_data(cls, *, fundamental, data_dict: dict[str, np.ndarray]):
        x_A, x_B = data_dict["x1"][:], data_dict["x2"][:]
        K_A, K_B = data_dict["K1"][:], data_dict["K2"][:]
        if fundamental.fundamental_matrix is None:
            return cls(K_A=K_A, K_B=K_B, rotation=None, translation=None)
        essential_matrix = K_B.mT @ fundamental @ K_A
        try:
            _, R, t, good = cv2.recoverPose(
                essential_matrix,
                calibrate_points(x_A, K_A),
                calibrate_points(x_B, K_B),
            )
        except cv2.error as _:
            R = np.eye(3)
            t = np.zeros(3)
        return cls(K_A=K_A, K_B=K_B, rotation=R, translation=t)

    @property
    def essential_matrix(self) -> np.ndarray:
        return skew_matrix(self.translation) @ self.rotation  # 3 x 3
    
    def calibrate(self, data: dict[str, np.ndarray]):
        self.K_A = data['K1'][:]
        self.K_B = data['K2'][:]
        return self
    
    def compute_squared_residuals(self, data_dict: dict[str, np.ndarray], *, subset:np.ndarray = None):
        if self.essential_matrix is None:
            return None
        if subset is None:
            x_A, x_B = data_dict['x1'][:], data_dict['x2'][:]
        else:
            x_A, x_B = data_dict['x1'][:][subset], data_dict['x2'][:][subset]

        calib = (
            1
            / 2
            * (
                1 / np.mean(self.K_A[[0, 1], [0, 1]])
                + 1 / np.mean(self.K_B[[0, 1], [0, 1]])
            )
        )
        # computes squared sampson error
        x_A, x_B = to_homog(x_A), to_homog(x_B)  # N x 3
        x_A, x_B = (
            x_A @ np.linalg.inv(self.K_A).mT,
            x_B @ np.linalg.inv(self.K_B).mT,
        )  # N x 3

        essential_matrix = self.essential_matrix  # 3 x 3
        Fx_A = essential_matrix @ x_A.mT  # 3 x N
        FTx_B = essential_matrix.mT @ x_B.mT  # 3 x N
        x_BFx_A = np.sum(x_B * (Fx_A.mT), axis=1)  # N
        denom = Fx_A[0] ** 2 + Fx_A[1] ** 2 + FTx_B[0] ** 2 + FTx_B[1] ** 2  # N
        calib_squared_residuals = x_BFx_A**2 / denom  # N
        pixel_squared_residuals = calib_squared_residuals / calib**2
        return pixel_squared_residuals

    def __array__(self):
        return self.essential_matrix


class Fundamental(GeometricModel):
    def __init__(
        self,
        *,
        fundamental_matrix: np.ndarray = None,
    ):
        super().__init__()
        self.fundamental_matrix = fundamental_matrix

    def compute_squared_residuals(self, data_dict: dict[str, np.ndarray], *, subset:np.ndarray = None):
        if subset is None:
            x_A, x_B = data_dict['x1'][:], data_dict['x2'][:]
        else:
            x_A, x_B = data_dict['x1'][:][subset], data_dict['x2'][:][subset]
        if self.fundamental_matrix is None:
            return None

        x_A, x_B = to_homog(x_A), to_homog(x_B)  # N x 3
        fundamental_matrix = self.fundamental_matrix  # 3 x 3
        Fx_A = fundamental_matrix @ x_A.mT  # 3 x N
        FTx_B = fundamental_matrix.mT @ x_B.mT  # 3 x N
        x_BFx_A = np.sum(x_B * (Fx_A.mT), axis=1)  # N
        denom = Fx_A[0] ** 2 + Fx_A[1] ** 2 + FTx_B[0] ** 2 + FTx_B[1] ** 2  # N
        return x_BFx_A**2 / denom  # N

    def __array__(self):
        return self.fundamental_matrix

    def calibrate(self, data_dict: dict[str, np.ndarray]):
        return Essential.from_fundamental_and_data(fundamental=self, data_dict = data_dict)
