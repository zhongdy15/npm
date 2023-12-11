import os
import time
envs = ['Pusher-v2']
config_name = 'pusher'
alg = 'PPO'
n_repeats = 5

for trials in range(n_repeats):
    for env in envs:
        cmd_line = f"python -m pureppo.train " \
                   f" --f pureppo/config/{config_name} " \
                   f" --algorithm_type {alg} &"
        print(cmd_line)
        os.system(cmd_line)
        time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
