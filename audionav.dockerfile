FROM soundspaces

ADD agent.py agent.py
ADD ppo_agent.py ppo_agent.py
ADD submission.sh submission.sh
ADD configs/challenge_audionav.local.rgbd.yaml challenge_audionav.local.rgbd.yaml
ADD configs/ppo_audionav.yaml ppo_audionav.yaml
ENV AGENT_EVALUATION_TYPE remote

ENV TRACK_CONFIG_FILE "challenge_audionav.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate soundspaces && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh"]
