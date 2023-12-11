import os
import time

all_envs =  ['GoToRedBallGreyAR-v0',"GoToDoorOpenR2AR-v0","GoToDoorOpenR2AddPositionBonus-v0",
             "GoToDoorOpenR2GreyKeyAR-v0",'GoToDoorOpenR2GreyKeyAddPositionBonus-v0',
             "GoToDoorOpenR2GreenBoxAR-v0","GoToDoorOpenR2GreenBoxAddPositionBonus-v0",
             "GoToLocalAR-v0",
             "GoToR3BlueBallAddPositionBonus-v0","GoToR3PurpleBallAddPositionBonus-v0","GoToR3BlueKeyAddPositionBonus-v0",
             "GoToR3GreyKeyAddPositionBonus-v0","GoToR3GreenBoxAddPositionBonus-v0","GoToR3RedBoxAddPositionBonus-v0",
             "PutNextLocalAR-v0",]


envs = ["GoToR3BlueKeyAddPositionBonus-v0","GoToR3GreenBoxAddPositionBonus-v0","GoToR3PurpleBallAddPositionBonus-v0",]

config_name = 'babyaiar'

alg = 'PPO'
n_repeats = 4
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
                       f" --env_id {env}" \
                       f" --mask {mask_flag} " \
                       f" --mask_threshold {mask_threshold} " \
                       f" --total_timesteps {total_timesteps} &"
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
