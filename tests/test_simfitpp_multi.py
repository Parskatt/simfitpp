from tqdm import tqdm
from simfitpp import SIMFITPP
from simfitpp.geom_estimators import PoseLibFundamental, PoseLibEssential, MAGSACOpenCVFundamental
import numpy as np
import h5py

from simfitpp.utils import pose_error, compute_auc

if __name__ == "__main__":
    dataset = h5py.File("data/relative/scannet1500_spsg.h5")
    th_guess = 1.
    geom_estimators = [PoseLibFundamental(), PoseLibEssential(), MAGSACOpenCVFundamental()]
    for geom_estimator in geom_estimators:
        threshold_estimator = SIMFITPP(
            geom_estimator=geom_estimator,
            train_fraction=0.5,
            max_iter=4)
        th_ests = []
        successes = []
        errs = []
        errs = []
        calib_model = ()
        th_est = th_guess
        for name, data in tqdm(dataset.items()):
            th_est, success = threshold_estimator(data_dict = data, th_guess = th_est, num_previous_est=sum(successes))
            geom_model = geom_estimator(data, th_est)
            calib_geom_model = geom_model.calibrate(data)
            err = pose_error(data['R'][:], data['t'][:], calib_geom_model.rotation, calib_geom_model.translation)    
            th_ests.append(th_est)
            successes.append(success)
            errs.append(err)
        errs = np.array(errs)
        print(compute_auc(errs, [5., 10., 20.]))
