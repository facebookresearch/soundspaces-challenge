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
from av_nav.config.default import get_task_config


class RandomAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS

    def reset(self):
        pass

    def act(self, observations):
        return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    args = parser.parse_args()

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = get_task_config(config_paths)
    agent = RandomAgent(task_config=config)

    if args.evaluation == "local":
        challenge = soundspaces.Challenge(eval_remote=False)
    else:
        challenge = soundspaces.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
