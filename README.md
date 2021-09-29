# NeurIPS 2021 IGLU competition Dreamer baseline. 

This repository contains an implementation of the Dreamer baseline for the Builder task of the IGLU contest. The code is fully submittable; it contains scripts with a submission-ready agent and for testing a submittable agent.

This baseline provides an implementation of [DreamerV2](https://arxiv.org/abs/2010.02193) model-based RL algorithm for the iglu env.

# Code structure

In `dreamerv2/`, you will find the main files for defining and training the agent. In `common/` there are utility classes and functions for the main code. Environment class alongside its wrappers are defined in `common/envs.py`.
File `one_episode/one_episode-500.npz` defines input shapes for the model in evaluation.

File `custom_agent.py` defines a `CusomAgent` class that can load a pretrained Dreamer model and predict actions given observations from the env. `test_submission.py` holds the code that can test the submission code locally. We recommend you test your submission for bugs as the script contains effectively the same code as in the evaluation server.

# Performance 

This section will be updated with numerical results soon. 

The current baseline works in the single-task setting. 

The Dreamer acts given a visual input only. In the current state, it can solve simple tasks, e.g., five blocks L structure C32, and more complex multicolor tasks, e.g., 12 blocks table of two colors, task C8.

Surprisingly, given a grid and position in the input, the model performs much worse.

# Launch requirements

This model is GPU intensive but sample efficient as it requires just one instance of the env to be trained in a reasonable time.
The model uses around 5GB of GPU memory and 35 GB of RAM to train. The model uses mixed-precision compute, so to get a similar time performance, you should use the corresponding GPU. 

For task C32, the model converges in 150-200k env steps or 3-4 hours. For task C8, the model converges in 2M steps or around a day. These numbers were obtained on a machine with v100, a 2.3 GHz CPU, and an HDD.

The main bottleneck except the GPU is the disk and the CPU because the model intensively reads a random experience from the whole history.

# Running the code

We highly recommend running everything from the docker container. Please, use pre-built image `iglucontest/baselines:builder-dreamer`.

To run the training, use the following command:

```
python dreamerv2/train.py --configs defaults iglu --logdir "./logs/$(date +%Y%m%d-%H%M%S)" --task iglu_C8
```

# How to submit a model

Before making a submission, make sure to test your code locally.
To test the submission, provide a valid path to a saved checkpoint in `custom_agent.py` and run

```
python test_submission.py
```

If you want your image to be used in the evaluation, put the name of your image into `Dockerimage`. Also, you should install the following mandatory **PYTHON PACKAGES** that are used in the evaluation server:
`docker==5.0.2 requests==2.26.0 tqdm==4.62.3 pyyaml==5.4.1 gym==0.18.3`.

If you use non-default hyperparameters, update them in `dreamerv2/configs.yml` as CustomAgent uses them
to build the model. Also, as `one_episode/one_episode-500.npz` defines input shapes for the model, you should update it also in case you tweak the inputs of your model.

To submit the model, just zip the whole directory (do not forget to add recursive paths) and submit the zip file in the [codalab competition](https://competitions.codalab.org/competitions/33828).

# Summary of changes

The baseline implemented with a number of changes in the model and the env. They are summarized below: 

Environment wrappers:

  * We unify block selection actions and placement actions. That is, each Hotbar selection will be immediately followed by the placement of the chosen block.
  * After each place/break action, we do **exactly** 3 noop actions to wait until the, e.g., break event will happen. This is due to how Minecraft processes actions internally.
  * The actions are discretized into a simple categorical space. There are four movement actions (forth, back, left, right); 4 camera actions (look up, look down, look left or right); each camera action changes one angle by 5 degrees. Jump action, break action, 6 Hotbar selection/placement actions, and one noop action. In total, there are 17 discrete actions.
  * We change the reward function; the new one tracks the maximal intersection size of two zones and gives only `+1` iff it increased a maximum reached size during the episode. This means that the maximal possible episode return is equal to the size of the target structure.

Model changes (notation follows the paper):

  * We use the same hyperparameters that were used to obtain the results for Atari. 
  * We train the model to predict the position of the agent and the current grid. We do so even if they are not fed as inputs (i.e., the model should predict them from the visual and dynamics information). At the test time, we still need only a visual signal as input.
  * We initialize the deterministic/stochastic state of RSSM with a processed position and grid state preceding the first state in the sequence chunk that is used to train the model.
