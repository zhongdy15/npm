# No Prior Mask: Eliminate Redundant Action for Deep Reinforcement Learning

## Requirements
We assume you have access to a gpu that can run CUDA 11.6. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
After the installation ends you can activate your environment with:
```
source activate npm
```
## Instructions 
### Phase 1: Training N-value network
To train similarity factor model on the `GoToR3` task from image-based observations  run:

```
python -m min_red.train \
    --f min_red/config/babyaiar \
    --algorithm_type PPO \
    --env_id GoToPositionBonus-v0 \
    --method Nill \
    --algorithm.learn.log_interval 10 \
    --total_timesteps 5000000
```

Run ***grid_search_\*.py***  will excute multiple commands at the same time.
```
python min_red/grid_search_babyai.py
```

This will produce 'log' folder, where all the outputs are going to be stored including N-value network(***mfmodel***). One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir log
```
and opening up tensorboad in your browser.

### Phase 2: Utilizing N-value network
Add the path of N-value network(***mfmodel***) stored above in ***makppo/train.py/Env_mask_dict***, run
```
python maskppo/grid_search_babyai.py
```
This will produce 'log' folder, where all the outputs are going to be stored.  

