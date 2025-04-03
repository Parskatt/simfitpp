import numpy as np
from simfitpp import SIMFITPP, PoseLibFundamental

N = 1000
D = 2
x_A = 100*np.random.randn(N, D)
x_B = 100*np.random.randn(N, D)
th_guess = 1.
geom_estimator = PoseLibFundamental(
    min_iterations=500,
    max_iterations=500,
    success_prob=0.9999
)
th_estimator = SIMFITPP(
    geom_estimator,
    alpha = 0.99, 
    max_iter = 4, 
    ftol = 0.01,
    train_fraction = 0.5,
    th_min = 0.25,
    th_max = 8.)
th_est, is_success = th_estimator.estimate_threshold(x_A, x_B, th_guess)
