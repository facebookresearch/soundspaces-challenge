<p align="center">
  <img width = "50%" src='res/img/soundspaces-logo.png' />
  </p>

--------------------------------------------------------------------------------

# SoundSpaces Challenge 2021

This repository contains starter code for the 2021 challenge, details of the tasks, and training and evaluation setups. For an overview of SoundSpaces Challenge visit [soundspaces.org/challenge](https://soundspaces.org/challenge/). 

This year, we are hosting challenges on audio-visual navigation task [1], where an agent is tasked to find a sound-making object in unmapped 3D environments with visual and auditory perception.


## AudioNav Task
In AudioGoal navigation (AudioNav), an agent is spawned at a random starting position and orientation in an unseen environment. A sound-emitting object is also randomly spawned at a location in the same environment. The agent receives a one-second audio input in the form of a waveform at each time step and needs to navigate to the target location. No ground-truth map is available and the agent must only use its sensory input (audio and RGB-D) to navigate.

### Dataset
The challenge will be conducted on the <a href="https://github.com/facebookresearch/sound-spaces/blob/master/soundspaces/README.md">SoundSpaces Dataset</a>, which is based on <a href="https://aihabitat.org/">AI Habitat</a>, <a href="https://niessner.github.io/Matterport//">Matterport3D</a>, and <a href="https://github.com/facebookresearch/Replica-Dataset">Replica</a>. For this challenge, we use the Matterport3D dataset due to its diversity and scale of environments. This challenge focuses on evaluating agents' ability to generalize to unheard sounds and unseen environments. The training and validation splits are the same as used in <i>Unheard Sound</i> experiments reported in the <a href="http://vision.cs.utexas.edu/projects/audio_visual_navigation/">SoundSpaces paper</a>. They can be downloaded from the <a href="https://github.com/facebookresearch/sound-spaces/tree/master/soundspaces">SoundSpaces repo</a>. For the challenge test split, we will use new sounds that are not currently publicly available on the website. 

### Evaluation
After calling the STOP action, the agent is evaluated using the 'Success weighted by Path Length' (SPL) metric [2]. 

<p align="center">
  <img src='res/img/spl.png' />
</p>


An episode is deemed successful if on calling the STOP action, the agent is within 0.36m (2x agent-radius) of the goal position.


## Participation Guidelines

Participate in the contest by registering on the [EvalAI challenge page](https://evalai.cloudcv.org/web/challenges/challenge-page/806/overview) and creating a team. Participants will upload docker containers with their agents that evaluated on a AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to make sure it is working. Instructions for training, local evaluation, and online submission are provided below.

### Local Evaluation

1. Clone the challenge repository:  

    ```bash
    git clone https://github.com/changanvr/soundspaces-challenge.git
    cd soundspaces-challenge
    ```

1. Implement your own agent or try one of ours. We provide an agent in `agent.py` that takes random actions:
    ```python
    import habitat
    import soundspaces

    class RandomAgent(habitat.Agent):
        def reset(self):
            pass

        def act(self, observations):
            return {"action": numpy.random.choice(task_config.TASK.POSSIBLE_ACTIONS)}

    def main():
        agent = RandomAgent(task_config=config)
        challenge = soundspaces.Challenge()
        challenge.submit(agent)
    ```
    [Optional] Modify submission.sh file if your agent needs any custom modifications (e.g. command-line arguments). Otherwise, nothing to do. Default submission.sh is simply a call to `RandomAgent` agent in `agent.py`. 


1. Install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) following instructions here: [https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)). 
Note: only supports Linux; no Windows or MacOS.

1. Modify the provided Dockerfile if you need custom modifications. Let's say your code needs `pytorch`, these dependencies should be pip installed inside a conda environment called `soundspaces` that is shipped with our soundspaces/challenge docker, as shown below:

    ```dockerfile
    FROM soundspaces/challenge:2021

    # install dependencies in the soundspaces conda environment
    RUN /bin/bash -c ". activate soundspaces; pip install torch"

    ADD agent.py /agent.py
    ADD submission.sh /submission.sh
    ```
    Build your docker container: `docker build . --file audionav.dockerfile  -t audionav_submission` or using `docker build . --file audionav.dockerfile  -t audionav_submission`. (Note: you may need `sudo` priviliges to run this command.)

2. Following instructions for downloading SoundSpaces [dataset](https://github.com/facebookresearch/sound-spaces/tree/master/soundspaces) and place all data under data/ folder.

    **Using Symlinks:**  If you used symlinks (i.e. `ln -s`) to link to an existing downloaded data, there is an additional step. Make sure there is only one level of symlink (instead of a symlink to a symlink link to a .... symlink) with
      ```bash
      ln -f -s $(realpath data/scene_datasets/mp3d) \
          data/scene_datasets/mp3d
      ```
     
     <!-- Then modify the docker command `test_locally_audionav_rgbd` to mount the linked to location by adding 
     `-v $(realpath habitat-challenge-data/data/scene_datasets/mp3d)`.  The modified docker command
     would be
     ```bash
      docker run \
          -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
          -v $(realpath habitat-challenge-data/data/scene_datasets/mp3d) \
          --runtime=nvidia \
          -e "AGENT_EVALUATION_TYPE=local" \
          -e "TRACK_CONFIG_FILE=/challenge_objectnav2020.local.rgbd.yaml" \
          ${DOCKER_NAME}
    ``` -->

3. Evaluate your docker container locally:
    ```bash
    # Testing AudioNav
    ./test_locally_audionav_rgbd.sh --docker-name audionav_submission
    ```
    If the above command runs successfully you will get an output similar to:
    ```
    2019-02-14 21:23:51,798 initializing sim Sim-v0
    2019-02-14 21:23:52,820 initializing task Nav-v0
    2020-02-14 21:23:56,339 distance_to_goal: 5.205519378185272
    2020-02-14 21:23:56,339 spl: 0.0
    ```
    Note: this same command will be run to evaluate your agent for the leaderboard. **Please submit your docker for remote evaluation (below) only if it runs successfully on your local setup.**  

### Online submission

Follow instructions in the `submit` tab of the EvalAI challenge page (coming soon) to submit your docker image. Note that you will need a version of EvalAI `>= 1.3.5`. Pasting those instructions here for convenience:

```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.3.5"

# Set EvalAI account token
evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
evalai push audionav_submission:latest --phase <phase-name>
```

Valid challenge phases are `soundspaces21-audionav-{minival, test-std, test-ch}`.

The challenge consists of the following phases:

1. **Minival phase**: This split is same as the one used in `./test_locally_audionav_rgbd.sh`. The purpose of this phase/split is sanity checking -- to confirm that our remote evaluation reports the same result as the one you're seeing locally. Each team is allowed maximum of 30 submission per day for this phase, but please use them judiciously. We will block and disqualify teams that spam our servers. 
2. **Test Standard phase**: The purpose of this phase/split is to serve as the public leaderboard establishing the state of the art; this is what should be used to report results in papers. Each team is allowed maximum of 10 submission per day for this phase, but again, please use them judiciously. Don't overfit to the test set. 
3. **Test Challenge phase**: This phase/split will be used to decide challenge winners. Each team is allowed total of 5 submissions until the end of challenge submission phase. The highest performing of these 5 will be automatically chosen. Results on this split will not be made public until the announcement of final results at the [Embodied AI workshop at CVPR](https://embodied-ai.org/). 

Note: Your agent will be evaluated on 1000-2000 episodes and will have a total available time of 24 hours to finish. Your submissions will be evaluated on AWS EC2 p2.xlarge instance which has a Tesla K80 GPU (12 GB Memory), 4 CPU cores, and 61 GB RAM. If you need more time/resources for evaluation of your submission please get in touch. If you face any issues or have questions you can ask them by opening an issue on this repository.

### AudioNav Baselines and Starter Code
We have added a config in `configs/ppo_pointnav.yaml` that includes the av-nav baseline using PPO from SoundSpaces. 

<!-- 1. Install the [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/) packages. Also ensure that habitat-baselines is installed when installing Habitat-Lab by installing it with ```python setup.py develop --all```

2. Download the Gibson dataset following the instructions [here](https://github.com/StanfordVL/GibsonEnv#database). After downloading extract the dataset to folder `habitat-challenge/habitat-challenge-data/data/scene_datasets/gibson/` folder (this folder should contain the `.glb` files from gibson). Note that the `habitat-lab` folder is the [habitat-lab](https://github.com/facebookresearch/habitat-lab/) repository folder. The data also needs to be in the habitat-challenge-data/ in this repository.

3. **Pointnav**: Download the dataset for Gibson PointNav from [link](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip) and place it in the folder `habitat-challenge/habitat-challenge-data/data/datasets/pointnav/gibson`. If placed correctly, you should have the train and val splits at `habitat-challenge/habitat-challenge-data/data/datasets/pointnav/gibson/v2/train/` and `habitat-challenge/habitat-challenge-data/data/datasets/pointnav/gibson/v2/val/` respectively. Place Gibson scenes downloaded in step-4 of local-evaluation under the `habitat-challenge/habitat-challenge-data/data/scene_datasets` folder. If you have already downloaded thes files for the habitat-lab repo, you may simply symlink them using `ln -s $PATH_TO_SCENE_DATASETS habitat-challenge-data/data/scene_datasets` (if on OSX or Linux).

    **Objectnav**: Download the episodes dataset for Matterport3D ObjectNav from [link](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip) and place it in the folder `habitat-challenge/habitat-challenge-data/data/datasets/objectnav/mp3d`. If placed correctly, you should have the train and val splits at `habitat-challenge/habitat-challenge-data/data/datasets/objectnav/mp3d/v1/train/` and `habitat-challenge/habitat-challenge-data/data/datasets/objectnav/mp3d/v1/val/` respectively. Place Gibson scenes downloaded in step-4 of local-evaluation under the `habitat-challenge/habitat-challenge-data/data/scene_datasets` folder. If you have already downloaded thes files for the habitat-lab repo, you may simply symlink them using `ln -s $PATH_TO_SCENE_DATASETS habitat-challenge-data/data/scene_datasets` (if on OSX or Linux).

4. An example on how to train DD-PPO model can be found in [habitat-lab/habitat_baselines/rl/ddppo](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines/rl/ddppo). See the corresponding README in habitat-lab for how to adjust the various hyperparameters, save locations, visual encoders and other features. 
    
    1. To run on a single machine use the following script from `habitat-lab` directory, where `$task={pointnav, objectnav}`:
        ```bash
        #/bin/bash

        export GLOG_minloglevel=2
        export MAGNUM_LOG=quiet

        set -x
        python -u -m torch.distributed.launch \
            --use_env \
            --nproc_per_node 1 \
            habitat_baselines/run.py \
            --exp-config configs/ddppo_${task}.yaml \
            --run-type train \
            TASK_CONFIG.DATASET.SPLIT 'train' 
        ```
    2. There is also an example of running the code distributed on a cluster with SLURM. While this is not necessary, if you have access to a cluster, it can significantly speed up training. To run on multiple machines in a SLURM cluster run the following script: change ```#SBATCH --nodes $NUM_OF_MACHINES``` to the number of machines and ```#SBATCH --ntasks-per-node $NUM_OF_GPUS``` and ```$SBATCH --gres $NUM_OF_GPUS``` to specify the number of GPUS to use per requested machine.
        ```bash
        #!/bin/bash
        #SBATCH --job-name=ddppo
        #SBATCH --output=logs.ddppo.out
        #SBATCH --error=logs.ddppo.err
        #SBATCH --gres gpu:1
        #SBATCH --nodes 1
        #SBATCH --cpus-per-task 10
        #SBATCH --ntasks-per-node 1
        #SBATCH --mem=60GB
        #SBATCH --time=12:00
        #SBATCH --signal=USR1@600
        #SBATCH --partition=dev

        export GLOG_minloglevel=2
        export MAGNUM_LOG=quiet

        export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

        set -x
        srun python -u -m habitat_baselines.run \
            --exp-config configs/ddppo_${task}.yaml \
            --run-type train \
            TASK_CONFIG.DATASET.SPLIT 'train' 
        ```
    3. **Notes about performance**: We have noticed that turning on the RGB/Depth sensor noise may lead to reduced simulation speed. As such, we recommend initially training with these noises turned off and using them for fine tuning if necessary. This can be done by commenting out the lines that include the key "NOISE_MODEL" in the config: ```habitat-challenge/configs/challenge_pointnav2020.local.rgbd.yaml```.
    4. The preceding two scripts are based off ones found in the [habitat_baselines/ddppo](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines/rl/ddppo).

5. The checkpoint specified by ```$PATH_TO_CHECKPOINT ``` can evaluated by SPL and other measurements by running the following command:

    ```bash
    python -u -m habitat_baselines.run \
        --exp-config configs/ddppo_${task}.yaml \
        --run-type eval \
        EVAL_CKPT_PATH_DIR $PATH_TO_CHECKPOINT \
        TASK_CONFIG.DATASET.SPLIT val
    ```
    The weights used for our DD-PPO Pointnav or Objectnav baseline for the Habitat-2020 challenge can be downloaded with the following command:
    ```bash
    wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo_${task}_habitat2020_challenge_baseline_v1.pth
    ```, where `$Task={pointnav, objectnav}.

    The default *Pointnav* DD-PPO baseline is trained for 120 Updates on 10 million frames with the config param: ```RL.SLACK_REWARD '-0.001'``` which reduces the slack reward to -0.001.
    The default *Objectnav* DD-PPO baseline is trained for 266 Updates on 209 million frames  with the provided config.

6. To submit your entry via EvalAI, you will need to build a docker file. We provide Dockerfiles ready to use with the DD-PPO baselines in `${Task}_DDPPO_baseline.Dockerfile`, where `$Task={Pointnav, Objectnav}`. For the sake of completeness, we describe how you can make your own Dockerfile below. If you just want to test the baseline code, feel free to skip this bullet because  ```${Task}_DDPPO_baseline.Dockerfile``` is ready to use.
    1. You may want to modify the `${Task}_DDPPO_baseline.Dockerfile` to include PyTorch or other libraries. To install pytorch, ifcfg and tensorboard, add the following command to the Docker file:
        ```dockerfile
        RUN /bin/bash -c ". activate habitat; pip install ifcfg torch tensorboard"
        ```
    2. You change which ```agent.py``` and which ``submission.sh`` script is used in the Docker, modify the following lines and replace the first agent.py or submission.sh with your new files.:
        ```dockerfile
        ADD agent.py agent.py
        ADD submission.sh submission.sh
        ```
    3. Do not forget to add any other files you may need in the Docker, for example, we add the ```demo.ckpt.pth``` file which is the saved weights from the DD-PPO example code.
    
    4. Finally modify the submission.sh script to run the appropiate command to test your agents. The scaffold for this code can be found ```agent.py``` and the DD-PPO specific agent can be found in ```ddppo_agents.py```. In this example, we only modify the final command of the PointNav/ObjectNav docker: by adding the following args to submission.sh ```--model-path demo.ckpt.pth --input-type rgbd```. The default submission.sh script will pass these args to the python script. You may also replace the submission.sh. 
        1. Please note that at this time, that habitat_baselines uses a slightly different config system and the configs nodes for habitat are defined under TASK_CONFIG which is loaded at runtime from BASE_TASK_CONFIG_PATH. We manually overwrite this config using the opts args in our agent.py file.

7. Once your Dockerfile and other code is modified to your satisfcation, build it with the following command.
    ```bash
    docker build . --file ${Task}_DDPPO_baseline.Dockerfile -t ${task}_submission
    ```
8. To test locally simple run the ```test_locally_${task}_rgbd.sh``` script. If the docker runs your code without errors, it should work on Eval-AI. The instructions for submitting the Docker to EvalAI are listed above.
9. Happy hacking! -->
<!-- 
## Citing Sound Challenge 2020
Please cite the following paper for details about the 2020 PointNav challenge:

```
@inproceedings{habitat2020sim2real,
  title     =     {Are We Making Real Progress in Simulated Environments? Measuring the Sim2Real Gap in Embodied Visual Navigation},
  author    =     {Abhishek Kadian, Joanne Truong, Aaron Gokaslan, Alexander Clegg, Erik Wijmans, Stefan Lee, Manolis Savva, Sonia Chernova, Dhruv Batra},
  booktitle =     {arXiv:1912.06321},
  year      =     {2019}
}
``` -->

## Acknowledgments

Thank Oleksandr Maksymets and Rishabh Jain for the technical support. And thank Habitat team for the challenge template.

<!-- The Habitat challenge would not have been possible without the infrastructure and support of [EvalAI](https://evalai.cloudcv.org/) team. We also thank the work behind [Gibson](http://gibsonenv.stanford.edu/) and [Matterport3D](https://niessner.github.io/Matterport/) datasets.  -->

## References

[1] [SoundSpaces: Audio-Visual Navigation in 3D Environments](https://arxiv.org/pdf/1912.11474.pdf). Changan Chen\*, Unnat Jain\*, Carl Schissler, Sebastia Vicenc Amengual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu, Philip Robinson, Kristen Grauman. ECCV, 2020.

[2] [On evaluation of embodied navigation agents](https://arxiv.org/abs/1807.06757). Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, Amir R. Zamir. arXiv:1807.06757, 2018.


## License
This repo is MIT licensed, as found in the LICENSE file.