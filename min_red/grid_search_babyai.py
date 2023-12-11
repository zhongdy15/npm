import os
import time

all_envs =  ['GoToRedBallGreyPositionBonus-v0','GoToR2PositionBonus-v0',"GoToDoorOpenR2PositionBonus-v0","GoToLocalPositionBonus-v0",
             "GoToPositionBonus-v0","KeyCorridorS5R2PositionBonus-v0","PutNextLocalPositionBonus-v0"]
index= 4

envs = [all_envs[index]]
config_name = 'babyaiar'
methods = ['Nill']
alg = 'PPO'
n_repeats = 3
total_timesteps = 5000000
log_interval = 10

for trials in range(n_repeats):
    for env in envs:
        for method in methods:
            cmd_line = f"python -m min_red.train " \
                       f" --f min_red/config/{config_name} " \
                       f" --algorithm_type {alg} " \
                       f" --env_id {env}" \
                       f" --method {method} " \
                       f" --algorithm.learn.log_interval {log_interval} " \
                       f" --total_timesteps {total_timesteps}" \
                       f" --wandb False & "
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(10)
