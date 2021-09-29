import tensorflow as tf
import gym
import sys

sys.path.insert(0, './dreamerv2')

import agent
import common
import ruamel.yaml as yaml
from copy import deepcopy as copy
import elements
import pathlib

class CustomAgent:
  def __init__(self, action_space):
    self.action_space = action_space
    self.get_discretization(self.action_space)
    configs = pathlib.Path('dreamerv2/configs.yaml')
    configs = yaml.safe_load(configs.read_text())
    config = elements.Config(configs['defaults'])
    config = config.update(configs['iglu'])
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))
    # just to init the model
    one_episode = pathlib.Path('./one_episode').expanduser()
    one_ep_replay = common.Replay(one_episode, config.replay_size, **config.replay)
    one_ep_data = iter(one_ep_replay.dataset(**config.dataset, config=config))
    self.agent = agent.Agent(config, logger=None, actspce=self.action_space, step=0, dataset=one_ep_data)
    # load weights
    self.agent.load('./c32.pkl')
    self.actions = iter([])
    self.state = None

  def policy(self, obs, reward, done, info, state):
    obs['reward'] = float(reward)
    obs['image'] = obs['pov']
    del obs['pov']
    # obs['grid'] = obs['grid'].flatten()
    obs['compass'] = obs['compass']['angle']
    # obs['position'] = obs['agentPos']
    obs['reset'] = done
    # del obs['agentPos']
    obs = tf.nest.map_structure(tf.convert_to_tensor, obs)
    obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), obs)
    action, state = self.agent.policy(obs, state, mode='eval')
    action = action['action'].numpy().argmax().item()
    return action, state

  def get_discretization(self, action_space, ):
    from minerl_patched.herobraine.hero.spaces import Discrete
    camera_delta = 5
    binary = ['attack', 'forward', 'back', 'left', 'right', 'jump']
    discretes = [action_space.no_op()]
    for op in binary:
      dummy = action_space.no_op()
      dummy[op] = 1
      discretes.append(dummy)
    camera_x = action_space.no_op()
    camera_x['camera'][0] = camera_delta
    discretes.append(camera_x)
    camera_x = action_space.no_op()
    camera_x['camera'][0] = -camera_delta
    discretes.append(camera_x)
    camera_y = action_space.no_op()
    camera_y['camera'][1] = camera_delta
    discretes.append(camera_y)
    camera_y = action_space.no_op()
    camera_y['camera'][1] = -camera_delta
    discretes.append(camera_y)
    for i in range(6):
      dummy = action_space.no_op()
      dummy['hotbar'] = i + 1
      discretes.append(dummy)
    discretes.append(action_space.no_op())
    self.discretes = discretes
    self.action_space = Discrete(len(discretes))
    self.old_action_space = action_space
    self.last_action = None

  def unwrap_step(self, action=None, raw_action=None):
    if action is not None:
      action = self.discretes[action]
    elif raw_action is not None:
      action = raw_action
    if action['hotbar'] != 0:
      yield action
      action = self.action_space.no_op()
      action = self.discretes[action]
      action['use'] = 1
    if action['use'] == 1 or action['attack'] == 1:
      # three empty actions, to make sure use/attack will take effect
      for _ in range(3):
        yield action
        action = self.action_space.no_op()
        action = self.discretes[action]
    yield action

  def act(self, obs, reward, done, info):
    if done:
      self.actions = iter([])
      self.state = None
      return
    try:
      action = next(self.actions)
    except StopIteration:
      agent_action, self.state = self.policy(obs, reward, done, info, self.state)
      self.actions = iter(self.unwrap_step(agent_action))
      action = next(self.actions)
    return copy(action)
