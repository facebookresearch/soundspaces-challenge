#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random

import numpy

import habitat
import soundspaces
# from av_nav.config.default import get_task_config
from ss_baselines.av_nav.config.default import get_task_config
from eval import Challenge2022


class RandomAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS

    def reset(self):
        pass

    def act(self, observations):
        return numpy.random.choice(len(self._POSSIBLE_ACTIONS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",  type=str, default="runs/",
    )
    args = parser.parse_args()

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = get_task_config(config_paths)
    agent = RandomAgent(task_config=config)

    challenge = Challenge2022()

    challenge.submit(agent, run_dir=args.run_dir, json_filename=f"random_{config.DATASET.SPLIT}.json")


if __name__ == "__main__":
    main()
