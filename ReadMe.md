# Bias-aware Adaptive Loss for Sequential Recommendation

<img width="3677" height="1439" alt="Image" src="https://github.com/user-attachments/assets/ae17c246-d964-41e8-ac40-5abb80319490" />


## Overview

Overview of the proposed method (BaSe). Given scores from any SR backbone and samples from a time-varying sampler $q(i \mid u,t)$, BaSe predicts a context-dependent correction factor $\alpha(u,t)$, applies adaptive log-$q$ debiasing, SNIPS reweighting, and calibration to produce debiased training signals.

## Dataset

The dataset can be found at the following link:

- https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ (Amazon)
- https://yelp.com/dataset (Yelp)

## Execution

- python --model [model_name].py --mode [idx] --dataset [dataset]
  - model {sasrec|bert4rec|duorec|core|srgnn|gcsan|gce-gnn|tagnn}
  - mode
    - idx = 0 (default) -> original model
    - idx = 1 -> IPS/SNIPS
    - idx = 2 -> BaSe
  - dataset
    - {toys|beauty|sports|yelp}
