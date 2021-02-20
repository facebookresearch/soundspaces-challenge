#!/usr/bin/env bash

# python agent.py --evaluation $AGENT_EVALUATION_TYPE $@

python avnav_agent.py --evaluation $AGENT_EVALUATION_TYPE --input-type depth --model-path example_ckpt.pth $@

