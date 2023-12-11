import os
import time

all_envs =  ['GoToRedBallGreyPositionBonus-v0','GoToR2PositionBonus-v0',"GoToDoorOpenR2PositionBonus-v0"]
index= -1

# choose index
index = 2
if index > -1 :
    envs = [all_envs[index]]
config_name = 'babyaiposbonus'

alg = 'PPO'
n_repeats = 5
total_timesteps = 1000000
log_interval = 10

# n_steps = 512

for trials in range(n_repeats):
    for env in envs:
        cmd_line = f"python -m pureppo.train " \
                   f" --f pureppo/config/{config_name} " \
                   f" --env_id {env}" \
                   f" --algorithm_type {alg} " \
                   f" --total_timesteps {total_timesteps} &"
        print(cmd_line)
        os.system(cmd_line)
        time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
