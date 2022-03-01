#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Optional

from habitat.core.logging import logger
from soundspaces.challenge import Challenge


EVAL_DCT_KEY_TO_METRIC_NAME = {"SPL": "spl", "SOFT_SPL": "softspl", "DISTANCE_TO_GOAL": "distance_to_goal", "SUCCESS": "success"}


class Challenge2022(Challenge):
    def __init__(self, eval_remote=False):
        config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
        super().__init__(eval_remote=eval_remote)

    def evaluate_custom(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)
        trajs: Dict = defaultdict(list)

        for count_episodes in tqdm(range(num_episodes)):
            agent.reset()
            observations = self._env.reset()

            scene_id = self._env.current_episode.scene_id.split("/")[-1].split(".")[0]
            episode_id = self._env.current_episode.episode_id
            scene_episode_id = f"{scene_id}_{episode_id}"

            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)

                if scene_episode_id not in trajs:
                    trajs[scene_episode_id] = [action]
                else:
                    trajs[scene_episode_id].append(action)

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                agg_metrics[m] += v

        avg_metrics = {k: v / (count_episodes + 1) for k, v in agg_metrics.items()}

        eval_dct = {"ACTIONS": trajs}
        for eval_dct_key in EVAL_DCT_KEY_TO_METRIC_NAME:
            assert EVAL_DCT_KEY_TO_METRIC_NAME[eval_dct_key] in avg_metrics
            eval_dct[eval_dct_key] = float(f"{avg_metrics[EVAL_DCT_KEY_TO_METRIC_NAME[eval_dct_key]]:.2f}")

        return avg_metrics, eval_dct

    def submit(self, agent, run_dir, json_filename):
        metrics, eval_dct = self.evaluate_custom(agent)

        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))

        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        with open(os.path.join(run_dir, json_filename), "w") as fo:
            json.dump(eval_dct, fo)
