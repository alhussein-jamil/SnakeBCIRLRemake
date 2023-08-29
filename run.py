# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from typing import Dict
import gym.spaces as spaces
import gym
import torch
import mediapy as media
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig
from rl_utils.common import (Evaluator, get_size_for_space,
                             set_seed)
from rl_utils.envs import create_vectorized_envs
from rl_utils.logging import Logger

from bcirl.imitation_learning.policy_opt.policy import Policy
from bcirl.imitation_learning.policy_opt.ppo import PPO
from bcirl.imitation_learning.policy_opt.storage import RolloutStorage
import yaml 
#register environment in gym
from gymnasium.envs.registration import register

def main(cfg) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    set_env_settings = {
        k: hydra_instantiate(v) if isinstance(v, DictConfig) else v
        for k, v in cfg.env.env_settings.items()
    }

    gym.register(
        id='snake',
        entry_point='environment.snake_env:SnakeEnv',
    )
    
    envs = create_vectorized_envs(
        cfg.env.env_name,
        cfg.num_envs,
        seed=cfg.seed,
        device=device,
        **set_env_settings,
    )


    steps_per_update = cfg.num_steps * cfg.num_envs
    num_updates = int(cfg.num_env_steps) // steps_per_update

    cfg.obs_shape = envs.observation_space.shape
    cfg.action_dim = get_size_for_space(envs.action_space)
    cfg.action_is_discrete = isinstance(cfg.action_dim, spaces.Discrete)
    cfg.total_num_updates = num_updates

    logger: Logger = hydra_instantiate(cfg.logger, full_cfg=cfg)

    storage: RolloutStorage = hydra_instantiate(cfg.storage, device=device)
    policy: Policy = hydra_instantiate(cfg.policy)
    policy = policy.to(device)
    updater = hydra_instantiate(cfg.policy_updater, policy=policy, device=device)
    
    start_update = 0
    if cfg.load_checkpoint is not None:
        ckpt = torch.load(cfg.load_checkpoint)
        updater.load_state_dict(ckpt["updater"], should_load_opt=cfg.resume_training)
        if cfg.load_policy:
            policy.load_state_dict(ckpt["policy"])
        if cfg.resume_training:
            start_update = ckpt["update_i"] + 1

    eval_info = {"run_name": logger.run_name}


    obs = envs.reset()
    storage.init_storage(obs)
    for update_i in range(start_update, num_updates):
        is_last_update = update_i == num_updates - 1
        for step_idx in range(cfg.num_steps):
            with torch.no_grad():
                act_data = policy.act(
                    storage.get_obs(step_idx),
                    storage.recurrent_hidden_states[step_idx],
                    storage.masks[step_idx],
                )
            next_obs, reward, done, info = envs.step(act_data["actions"])
            storage.insert(next_obs, reward, done, info, **act_data)
            logger.collect_env_step_info(info)

        updater.update(policy, storage, logger, envs=envs)

        storage.after_update()
        
        if True:
            
            # do a basic evaluation and render 
            snakie = gym.make('snakie-v0')
            snakie.reset()
            frames = []
            for i in range(100):
                snakie.render()
                action = policy.act(snakie.get_obs(), snakie.get_hidden(), snakie.get_mask())["actions"]
                obs, reward, done, info = snakie.step(action)
                frames.append(snakie.render(mode='rgb_array'))
                if done:
                    break
            snakie.close()
            media.write_video('snakie.mp4', frames, fps=10)
            
            
        if cfg.log_interval != -1 and (
            update_i % cfg.log_interval == 0 or is_last_update
        ):
            logger.interval_log(update_i, steps_per_update * (update_i + 1))

        if cfg.save_interval != -1 and (
            (update_i + 1) % cfg.save_interval == 0 or is_last_update
        ):
            save_name = osp.join(logger.save_path, f"ckpt.{update_i}.pth")
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "updater": updater.state_dict(),
                    "update_i": update_i,
                },
                save_name,
            )
            print(f"Saved to {save_name}")
            eval_info["last_ckpt"] = save_name

    logger.close()
    return eval_info


if __name__ == "__main__":
    cfg = yaml.load(open("bc-irl-snake.yaml", 'r'), Loader=yaml.SafeLoader)
    cfg = DictConfig(cfg)

    main(cfg)
