<p align="center">
  <h1 align="center"> Less Biased Noise Scale Estimation for Threshold-Robust RANSAC <br> IMW 2025</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=Ul-vMR0AAAAJ">Johan Edstedt</a>
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/abs/2503.07347](https://arxiv.org/abs/2503.13433" align="center">Paper</a>
  </p></h2>
</p>
<p align="center">
</p>
This repo provides code for estimating the inlier threshold for two-view relative pose estimation using RANSAC.


## Install
Assuming you have `uv` installed:
```bash
git clone git@github.com:Parskatt/simfitpp.git
cd simfitpp
uv sync
```


## API
Below is an example of the API to esimate the threshold for a pair of images.
```python
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
```

## Reproducing Results

### Disclaimer:

The code in this codebase is a reproduction of the internal code used for the paper.
As such there might be minor discrepencies.
I've checked that the code approximately reproduces SuperPoint + SuperGlue results on ScanNet-1500.
Note that due to randomness, the results may be slightly higher or lower than the paper results.

### To reproduce results:

```bash
bash scripts/download_data.sh
python tests/test_simfitpp.py
```
This should print out some AUC values.
The expected values should be around (+- 0.5)
```txt
[np.float64(0.14488472274986597), np.float64(0.28170116410278817), np.float64(0.4212129576093516)]
[np.float64(0.22573529158871225), np.float64(0.3982318903526437), np.float64(0.5513910056317942)]
[np.float64(0.13375165543509254), np.float64(0.2614397234088632), np.float64(0.39866589161181637)]
```

## Citing
```txt
@InProceedings{edstedt2025simfitpp,
    author    = {Edstedt, Johan},
    title     = {{Less Biased Noise Scale Estimation for Threshold-Robust RANSAC}},
    booktitle = {{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops}},
    month     = {June},
    year      = {2025}
}
```