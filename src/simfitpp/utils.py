import numpy as np


def to_homog(x):
    return np.concatenate((x, np.ones_like(x[..., -1:])), axis=-1)


def from_homog(x):
    return x[..., :-1] / x[..., -1:]


def skew_matrix(a: np.ndarray):
    zero = np.zeros_like(a[..., 0])
    a_x = np.array(
        [
            [zero, -a[..., 2], a[..., 1]],
            [a[..., 2], zero, -a[..., 0]],
            [-a[..., 1], a[..., 0], zero],
        ]
    )
    return a_x


def calib_matrix_to_camera_dict(K):
    camera_dict = {}
    camera_dict["model"] = "PINHOLE"
    camera_dict["width"] = int(np.ceil(K[0, 2] * 2))
    camera_dict["height"] = int(np.ceil(K[1, 2] * 2))
    camera_dict["params"] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    return camera_dict


def camera_dict_to_calib_matrix(cam):
    if cam["model"] == "PINHOLE":
        p = cam["params"]
        return np.array([[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])
    else:
        raise ValueError(
            f"Don't support {cam['model']} currently. Make a PR at https://github.com/Parskatt/simfitpp if you want it supported."
        )


def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def angle(v1, v2):
    v1s = np.squeeze(v1)
    v2s = np.squeeze(v2)
    n = np.linalg.norm(v1s) * np.linalg.norm(v2s)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1s, v2s) / n, -1.0, 1.0)))


def pose_error(R_gt, t_gt, R_est, t_est):
    if R_est is None or t_est is None:
        return 90.0
    rot_angle = rotation_angle(R_gt @ R_est.mT)
    trans_angle = angle(t_gt, t_est)
    return max(rot_angle, trans_angle)


def compute_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def calibrate_points(pts, K):
    pts_calib = pts.copy()
    pts_calib[:, 0] -= K[0, 2]
    pts_calib[:, 1] -= K[1, 2]
    pts_calib[:, 0] /= K[0, 0]
    pts_calib[:, 1] /= K[1, 1]
    return pts_calib
