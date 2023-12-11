import os
import time
envs = ['unlockpickupar-v0']
config_name = 'unlockpickupar'
alg = 'PPO'
n_repeats = 5
total_timesteps = 2000000
log_interval = 10
mask = ["True","False"]
mask_threshold = 1.0
# n_steps = 512

for trials in range(n_repeats):
    for mask_flag in mask:
        for env in envs:
            cmd_line = f"python -m maskppo.train " \
                       f" --f maskppo/config/{config_name} " \
                       f" --algorithm_type {alg} " \
                       f" --mask {mask_flag} " \
                       f" --mask_threshold {mask_threshold} " \
                       f" --total_timesteps {total_timesteps} &"
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
