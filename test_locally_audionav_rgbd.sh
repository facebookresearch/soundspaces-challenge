#!/usr/bin/env bash

DOCKER_NAME="audionav_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run \
    -v $(pwd)/data:/data \
    -v $(pwd)/data/binaural_rirs/mp3d:/data/binaural_rirs/mp3d \
    --runtime=nvidia \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=challenge_audionav.local.rgbd.yaml" \
    ${DOCKER_NAME}\
  