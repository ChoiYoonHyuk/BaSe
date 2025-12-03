# Bias-aware Adaptive Loss for Sequential Recommendation

<img width="3677" height="1439" alt="Image" src="https://github.com/user-attachments/assets/ae17c246-d964-41e8-ac40-5abb80319490" />


## Overview

We propose **Bias-aware Adaptive Loss for Sequential Recommendation (BaSe)**. 

## Dataset

We set the default dataset as "KuaiRand-27K.json", which can be found in the following link:

- https://kuairand.com/ (KuaiRand)
- https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ (Amazon)

## Execution

- python [model_name].py [idx]
  - idx = 0 (default) -> original model
  - idx = 1 -> IPS/SNIPS
  - idx = 2 -> BaSe
