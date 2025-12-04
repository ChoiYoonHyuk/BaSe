# Bias-aware Adaptive Loss for Sequential Recommendation

<img width="3677" height="1439" alt="Image" src="https://github.com/user-attachments/assets/ae17c246-d964-41e8-ac40-5abb80319490" />


## Overview

Overview of the proposed method (BaSe). Given scores from any SR backbone and samples from a time-varying sampler $q(i \mid u,t)$, BaSe predicts a context-dependent correction factor $\alpha(u,t)$, applies adaptive log-$q$ debiasing, SNIPS reweighting, and calibration to produce debiased training signals.

## Dataset

We set the default dataset as "KuaiRand-27K.json", which can be found in the following link:

- https://kuairand.com/ (KuaiRand)
- https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ (Amazon)

## Execution

python main.py \
  --dataset {toys|beauty|sports|yelp} \
  --model {sasrec|bert4rec|duorec|core|srgnn|gcsan|gce-gnn|tagnn} \
  --mode {0|1|2}
  
- python [model_name].py --mode [idx] --dataset [dataset]
  - mode
    - idx = 0 (default) -> original model
    - idx = 1 -> IPS/SNIPS
    - idx = 2 -> BaSe
  - dataset
    - beauty / toys / sports / yelp
