import os
import time
envs = ['UnlockToUnlockPositionBonus-v0']
config_name = 'unlocktounlockposbonus'
alg = 'PPO'
n_repeats = 5
total_timesteps = 1000000
log_interval = 10

# n_steps = 512

for trials in range(n_repeats):
    for env in envs:
        cmd_line = f"python -m pureppo.train " \
                   f" --f pureppo/config/{config_name} " \
                   f" --algorithm_type {alg} " \
                   f" --total_timesteps {total_timesteps} &"
        print(cmd_line)
        os.system(cmd_line)
        time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
