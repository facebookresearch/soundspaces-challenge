FROM soundspaces/challenge:2021

ADD avnav_agent.py avnav_agent.py
ADD configs/challenge_audionav.local.rgbd.yaml challenge_audionav.local.rgbd.yaml
ADD configs/avnav_agent.yaml avnav_agent.yaml
ADD example_ckpt.pth example_ckpt.pth

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "challenge_audionav.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate soundspaces && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && python avnav_agent.py --evaluation $AGENT_EVALUATION_TYPE --input-type depth --model-path example_ckpt.pth --wait-time 90"]