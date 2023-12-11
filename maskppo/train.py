import gym
import envs
import numpy as np
import time
import argparse
from common.format_string import pretty
from common.parser_args import get_config
from common.config import Config
import os
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy,CnnPolicy
from torch.nn.modules.activation import Tanh,ReLU
from stable_baselines3.common.evaluation import evaluate_policy
from common.evaluation import evaluate_policy_and_save
from .maskppo import MaskPPO
import wandb
torch.set_num_threads(8)

Env_mask_dict = {"unlockpickupar-v0":"Mask_unlockpickupar-v0_Nill_PPO_n-1_2023-04-13-12-25-45",
                 "GoToRedBallGreyAR-v0":"Mask_GoToRedBallGreyPositionBonus-v0_Nill_PPO_n-1_2023-04-27-19-34-27",
                 "GoToDoorOpenR2AR-v0":"Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2023-05-02-15-36-01",
                 "GoToDoorOpenR2AddPositionBonus-v0":"Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2023-05-02-15-36-01",
                 "GoToDoorOpenR2GreyKeyAR-v0":"Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2023-05-02-15-36-01",
                 'GoToDoorOpenR2GreyKeyAddPositionBonus-v0':"Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2023-05-02-15-36-01",
                 "GoToDoorOpenR2GreenBoxAR-v0":"Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2023-05-02-15-36-01",
                 "GoToDoorOpenR2GreenBoxAddPositionBonus-v0":"Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2023-05-02-15-36-01",
                 "GoToLocalAR-v0":"Mask_GoToLocalPositionBonus-v0_Nill_PPO_n-1_2023-05-04-17-50-23",
                 # GoToR3: different goal with the same mask
                 "GoToR3BlueBallAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_stochastic_PPO_n-1_2023-05-19-17-53-37",
                 "GoToR3PurpleBallAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_stochastic_PPO_n-1_2023-05-19-17-53-37",
                 "GoToR3BlueKeyAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_stochastic_PPO_n-1_2023-05-19-17-53-37",
                 "GoToR3GreyKeyAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_stochastic_PPO_n-1_2023-05-19-17-53-37",
                 "GoToR3GreenBoxAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_stochastic_PPO_n-1_2023-05-19-17-53-37",
                 "GoToR3RedBoxAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_stochastic_PPO_n-1_2023-05-19-17-53-37",
                 "PutNextLocalAR-v0":"Mask_PutNextLocalPositionBonus-v0_Nill_PPO_n-1_2023-05-08-16-51-54",
}


def train(config, log_path, mask_path, mask_flag, mask_threshold):
    if config.is_atari:
        make_env = make_atari_stack_env
    else:
        make_env = make_vec_env
    env = make_env(config.env_id, n_envs=1, vec_env_cls=DummyVecEnv,
                   vec_env_kwargs=config.vec_env_kwargs, env_kwargs=config.env_kwargs)

    if len(env.observation_space.shape) >=3:
        policy = 'CnnPolicy'
    else:
        policy = 'MlpPolicy'

    mf_model_path = os.path.join(mask_path, "mfmodel")
    from stable_baselines3.dqn.policies import MultiInputPolicy as ActionModel
    mf_model = ActionModel.load(mf_model_path,device=config.device)

    model = MaskPPO(policy, env, tensorboard_log=log_path, mf_model=mf_model, mask_flag=mask_flag,
                    mask_threshold=mask_threshold,**config.algorithm.policy)

    model.learn(**config.algorithm.learn)
    print("Finished training...")
    if config.save_model:
        print("Saving model...")
        model_path = os.path.join(log_path,"model")
        model.save(model_path)
        # test_mf_model = ActionModel.load(mf_model_path)
    if config.play_model:
        save_path = os.path.join(log_path,"eval.npy")
        mean, std = evaluate_policy_and_save(model, env, save_path=save_path, deterministic=False)
        print("mean:" + str(mean) + " std:" + str(std))


def bcast_config_vals(config):
    algorithm_config = Config(os.path.join(config.config_path, config.algorithm_type))
    config.merge({"algorithm": algorithm_config}, override=False)
    config.algorithm.learn.total_timesteps = config.total_timesteps
    config.algorithm.policy["device"] = config.device
    if "activation_fn" in config.algorithm.policy.policy_kwargs:
        activation_fn = config.algorithm.policy.policy_kwargs["activation_fn"]
        if activation_fn == "ReLU":
            config.algorithm.policy.policy_kwargs["activation_fn"] = ReLU
        elif activation_fn == "Tanh":
            config.algorithm.policy.policy_kwargs["activation_fn"] = Tanh
        else:
            raise NotImplementedError
    # config.algorithm.policy.method = config.method
    # config.algorithm.policy.wandb = config.wandb
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    parser.add_argument('--mask', type=str, default="False")
    parser.add_argument('--mask_threshold', type=float, default=1.0)
    args, extra_args = parser.parse_known_args()
    # get default parameters in this environment and override with extra_args
    config = get_config(args.f)
    # load default parameters in config_path/algorithm_type and override
    config = bcast_config_vals(config)
    pretty(config)

    if args.mask == "True":
        mask_flag = True
    else:
        mask_flag = False
    mask_threshold = args.mask_threshold

    goalstr = ''
    if 'goal' in config.env_kwargs.keys():
        goal = config.env_kwargs.goal
        goalstr = '_Goal'+str(goal)

    experiment_name = config.env_id + '_' +"mask"+ config.algorithm_type + goalstr + '_' + "Mask" + str(mask_flag) \
                      + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = os.path.join("log", experiment_name)

    # # use_mask_with_curiosity should be true
    # use_mask_with_curiosity = True
    #
    # if use_mask_with_curiosity:
    #     #mask_name = "Mask_unlockpickupar-v0_Nill_PPO_n-1_2023-03-16-17-18-18"
    #     mask_name = "Mask_unlockpickupar-v0_Nill_PPO_n-1_2023-04-13-12-25-45"
    # else:
    #     mask_name = "Mask_unlockpickupar-v0_Nill_PPO_n-1_2023-03-15-12-10-28"
    mask_name = Env_mask_dict[config.env_id]

    mask_path = os.path.join("log",mask_name)
    # if "wandb" in config:
    #     if config["wandb"]:
    #         # wandb.init(project="muzero-pytorch", entity='jyh', sync_tensorboard=True, name=args.exp_name,
    #         #            config=muzero_config.get_hparams(), dir=str(exp_path))
    #         wandb.init(
    #             # set the wandb project where this run will be logged
    #             project=config.env_id + '_' + config.algorithm_type,
    #
    #             # track hyperparameters and run metadata
    #             config=config.algorithm.policy.policy_kwargs,
    #             sync_tensorboard=True,
    #             name=experiment_name
    #             # dir=str(log_path)
    #         )
    train(config, log_path, mask_path, mask_flag, mask_threshold=mask_threshold)
    # wandb.finish()
