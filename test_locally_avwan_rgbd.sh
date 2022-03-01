#!/usr/bin/env bash

env CHALLENGE_CONFIG_FILE="configs/challenge_avwan.local.rgbd.yaml" python avwan_agent.py --input-type $INPUT_TYPE --model-path CKPT_NAME.pth$@
  