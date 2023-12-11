import os
import time
envs = ['rooms-v0']
config_name = 'rooms'
alg = 'PPO'
# methods = ['Nill']
n_repeats = 2#4
# abs_thresh = True
total_timesteps = 200000
# log_interval = 10  # (episodes)
#n_redundancies = 30
n_redundancy = [1,2,4,8,16,32]
n_steps = 100

for trials in range(n_repeats):
    for env in envs:
        # for method in methods:
        for n_redundancies in n_redundancy:
            cmd_line = f"python -m pureppo.train " \
                       f" --f pureppo/config/{config_name} " \
                       f" --algorithm_type {alg} " \
                       f" --total_timesteps {total_timesteps}" \
                       f" --env_kwargs.n_redundancies {n_redundancies} &"
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(10)
    #f" --algorithm.policy.n_steps {n_steps} " \
