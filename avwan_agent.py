#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
from collections import OrderedDict
import logging
import sys
import time

import numba
import numpy as np
import torch
from gym.spaces import Box, Dict, Discrete

import habitat
from habitat import Config
from habitat.core.agent import Agent
import soundspaces
from ss_baselines.av_wan.config.default import get_config
from ss_baselines.av_wan.ppo.policy import AudioNavBaselinePolicy
from ss_baselines.common.utils import batch_obs
from eval import Challenge2022


@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class PPOAgent(Agent):
    def __init__(self, config: Config):
        spaces = {
            "spectrogram": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(65, 26, 2),
                dtype=np.float32,
            ),
            "gm": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(400, 400, 2),
                dtype=np.float32,
            ),
            "am": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(20, 20, 1),
                dtype=np.float32,
            ),
            "action_map": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(9, 9, 1),
                dtype=np.float32,
            ),
        }

        if config.INPUT_TYPE in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(
                    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH,
                    1,
                ),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH,
                    3,
                ),
                dtype=np.uint8,
            )
        observation_spaces = Dict(spaces)

        action_space = Discrete(len(config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS))

        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        self.hidden_size = config.RL.PPO.hidden_size

        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        _seed_numba(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        policy_arguments = OrderedDict(
            observation_space=observation_spaces,
            hidden_size=self.hidden_size,
            goal_sensor_uuid=config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            masking=config.MASKING,
            action_map_size=9
        )

        self.actor_critic = AudioNavBaselinePolicy(**policy_arguments)
        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            print(f"Checkpoint loaded: {config.MODEL_PATH}")
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k.replace("actor_critic.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None

    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    def act(self, observations):
        batch = batch_obs([observations], device=self.device)

        with torch.no_grad():
            _, action, _, self.test_recurrent_hidden_states, _  = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(1.0)
            self.prev_actions.copy_(action)

        return action.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type", default="blind", choices=["blind", "rgb", "depth", "rgbd"]
    )
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument(
        "--run-dir",  type=str, default="runs/",
    )
    args = parser.parse_args()

    config = get_config(
        "configs/avwan_agent.yaml", ["BASE_TASK_CONFIG_PATH", config_paths]
    ).clone()
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path

    config.RANDOM_SEED = 7
    config.freeze()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    agent = PPOAgent(config)

    challenge = Challenge2022()
    challenge._env.seed(config.RANDOM_SEED)

    print("Start evaluating ...")
    challenge.submit(agent, run_dir=args.run_dir, json_filename=f"avwan_{config.TASK_CONFIG.DATASET.SPLIT}.json")


if __name__ == "__main__":
    main()