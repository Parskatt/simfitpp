from copy import copy
import cv2
import numpy as np
import poselib

from simfitpp.geom_models import Essential, Fundamental
from simfitpp.noise_models import Chi_1_2
from simfitpp.utils import calib_matrix_to_camera_dict
from .types import GeomEstimator


class PoseLibFundamental(GeomEstimator):
    def __init__(
        self,
        *,
        min_iterations: float = 500,
        max_iterations: float = 500,
        success_prob: float = .9999,
    ):
        super().__init__()
        self.opt = dict(
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            success_prob=success_prob,
        )
        self.model = Fundamental()
        self.noise_model = Chi_1_2()

    def __call__(self, data_dict: dict[str, np.ndarray],  th: float, *, subset: np.ndarray = None) -> Fundamental:
        x_A, x_B = self.load_corresps(data_dict, subset=subset)
        opt = copy(self.opt)
        opt["max_epipolar_error"] = th
        F, _ = poselib.estimate_fundamental(x_A, x_B, opt, {})
        self.model.fundamental_matrix = F
        return self.model


class PoseLibEssential(GeomEstimator):
    def __init__(
        self,
        *,
        min_iterations: float = 500,
        max_iterations: float = 500,
        success_prob: float = .9999,
    ):
        super().__init__()
        self.opt = dict(
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            success_prob=success_prob,
        )
        self.geom_model = Essential()
        self.noise_model = Chi_1_2()
    
    def __call__(self, data_dict: dict[str, np.ndarray], th: float, *, subset: np.ndarray = None) -> Essential:
        x_A, x_B = self.load_corresps(data_dict, subset=subset)
        self.geom_model.calibrate(data_dict)
        opt = copy(self.opt)
        opt["max_epipolar_error"] = th

        
        cam_A, cam_B = (
            calib_matrix_to_camera_dict(self.geom_model.K_A),
            calib_matrix_to_camera_dict(self.geom_model.K_B),
        )
        pose, _ = poselib.estimate_relative_pose(x_A, x_B, cam_A, cam_B, opt, {})
        self.geom_model.rotation, self.geom_model.translation = pose.R, pose.t
        return self.geom_model


class MAGSACOpenCVFundamental(GeomEstimator):
    def __init__(
        self,
        *,
        min_iterations: float = 500,
        max_iterations: float = 500,
        success_prob: float = .9999,
        refine: bool = True,
    ):
        super().__init__()
        self.min_iterations=min_iterations
        self.max_iterations=max_iterations
        self.success_prob=success_prob
        self.refine = refine
        self.model = Fundamental()
        self.noise_model = Chi_1_2()

    def __call__(self, data_dict: dict[str, np.ndarray],  th: float, *, subset: np.ndarray = None) -> Fundamental:
        x_A, x_B = self.load_corresps(data_dict, subset=subset)
        try:
            F, inl = cv2.findFundamentalMat(
                x_A,
                x_B,
                method=38, # ses lätt
                ransacReprojThreshold=th,
                confidence=self.success_prob,
                maxIters=self.max_iterations,
            )
        except:  # noqa: E722
            self.model.fundamental_matrix = None
            return self.model
        inl = inl.flatten().astype(bool)
        if not np.any(inl):
            self.model.fundamental_matrix = None
            return self.model
        if self.refine:
            F = refine_fundamental(x_A[inl], x_B[inl], F)
        self.model.fundamental_matrix = F
        return self.model


def refine_relative_pose(x1, x2, R, t, cam1, cam2):
    init_pose = poselib.CameraPose()
    init_pose.R = R
    init_pose.t = t
    pose, info = poselib.refine_relative_pose(x1, x2, init_pose, cam1, cam2, {})
    return pose.R, pose.t


def refine_fundamental(x1, x2, F):
    try:
        F, info = poselib.refine_fundamental(x1, x2, F, {})
    except:  # noqa: E722
        return None
    return F
