FROM soundspaces/challenge:2021

ADD avwan_agent.py avwan_agent.py
ADD configs/challenge_audionav.local.rgbd.yaml challenge_audionav.local.rgbd.yaml
ADD configs/avwan_agent.yaml avwan_agent.yaml
ADD av_wan.pth av_wan.pth

ENV AGENT_EVALUATION_TYPE remote
ENV USE_PLANNING_ENV true
ENV TRACK_CONFIG_FILE "challenge_audionav.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate soundspaces && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && python avwan_agent.py --evaluation $AGENT_EVALUATION_TYPE --input-type depth --model-path av_wan.pth --wait-time 90"]