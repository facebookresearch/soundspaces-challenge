#!/usr/bin/env bash

env CHALLENGE_CONFIG_FILE="configs/challenge_audionav.local.rgbd.yaml" python avnav_agent.py --input-type depth --model-path example_ckpt.pth
  