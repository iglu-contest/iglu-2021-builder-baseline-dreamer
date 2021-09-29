import os
import threading
from copy import deepcopy as copy

import gym
import numpy as np
from numpy.core.defchararray import lower


class Iglu:

  def __init__(
      self, name, action_repeat=1,
      asymmetric=False,
      action_space='human-level'):
    import iglu
    from iglu.tasks import TaskSet
    env = gym.make('IGLUSilentBuilder-v0', max_steps=5000, action_space='human-level')
    env.update_taskset(TaskSet(preset=[name]))
    env = ContDiscretization(env)
    if asymmetric:
      env = AsymmetricReward(env)
    self._env = env
    self.active_item = np.zeros(6, dtype=np.float32)

  @property
  def observation_space(self):
    grid_space = self._env.observation_space['grid']
    return gym.spaces.Dict({
        'image': self._env.observation_space['pov'],
        'inventory': self._env.observation_space['inventory'],
        'grid': gym.spaces.Box(low=0, high=7, shape=(np.prod(grid_space.shape),)),
        'position': self._env.observation_space['agentPos'],
        'compass': self._env.observation_space['compass']['angle'],
    })

  @property
  def action_space(self):
    return gym.spaces.Dict({'action': self._env.action_space})

  def close(self):
    return self._env.close()

  def reset(self):
    obs = self._env.reset()
    obs = {
      'image': obs['pov'], 
      'inventory': obs['inventory'], 
      'compass': obs['compass']['angle'],
      'grid': obs['grid'].flatten(),
      'position': obs['agentPos'],
    }
    return obs

  def step(self, action):
    action = action['action']
    obs, reward, done, info = self._env.step(action)
    # do not reward for removing wrong block
    # reward = 0 if reward == 1 else reward
    obs = {
      'image': obs['pov'], 
      'inventory': obs['inventory'], 
      'compass': obs['compass']['angle'],
      'grid': obs['grid'].flatten(),
      'position': obs['agentPos'],
    }
    return obs, reward, done, info


class AsymmetricReward(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.size = 0

  def reset(self):
    self.size = 0
    return super().reset()

  def step(self, action):
    obs, reward, done, info = super().step(action)
    intersection = self.env.unwrapped.task.task_monitor.max_int
    reward = max(intersection, self.size) - self.size
    self.size = max(intersection, self.size)
    return obs, reward, done, info


class ContDiscretization(gym.Wrapper):
  def __init__(self, env, ):
    super().__init__(env)
    camera_delta = 5
    binary = ['attack', 'forward', 'back', 'left', 'right', 'jump']
    discretes = [env.action_space.no_op()]
    for op in binary:
      dummy = env.action_space.no_op()
      dummy[op] = 1
      discretes.append(dummy)
    camera_x = env.action_space.no_op()
    camera_x['camera'][0] = camera_delta
    discretes.append(camera_x)
    camera_x = env.action_space.no_op()
    camera_x['camera'][0] = -camera_delta
    discretes.append(camera_x)
    camera_y = env.action_space.no_op()
    camera_y['camera'][1] = camera_delta
    discretes.append(camera_y)
    camera_y = env.action_space.no_op()
    camera_y['camera'][1] = -camera_delta
    discretes.append(camera_y)
    for i in range(6):
      dummy = env.action_space.no_op()
      dummy['hotbar'] = i + 1
      discretes.append(dummy)
    discretes.append(env.action_space.no_op())
    self.discretes = discretes
    self.action_space = gym.spaces.Discrete(len(discretes))
    self.old_action_space = env.action_space
    self.last_action = None

  def step(self, action=None, raw_action=None):
    if action is not None:
      action = self.discretes[action]
    elif raw_action is not None:
      action = raw_action
    if action['hotbar'] != 0:
      obs, hotbar_reward, done, info = self.env.step(action)
      if done:
        return obs, hotbar_reward, done, info
      action = self.env.action_space.noop()
      action['use'] = 1
    if action['use'] == 1 or action['attack'] == 1:
      total_reward = 0
      for i in range(3):
        obs, reward, done, info = self.env.step(action)
        total_reward += reward
        if done:
          break
        action = self.env.action_space.noop()
      return obs, total_reward, done, info
    return self.env.step(action)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class InventoryState(gym.Wrapper):
  def __init__(self, env) -> None:
    from minerl_patched.herobraine.hero.spaces import Dict
    from minerl_patched.herobraine.hero.spaces import Box
    self.env = env
    self.active_item = np.zeros(6, dtype=np.float32)
    self.active_item[0] = 1.
    observation_space = copy(env.observation_space.spaces)
    observation_space['item'] = Box(low=0., high=1., shape=(6,))
    self.observation_space = observation_space

  def step(self, action):
    if action['hotbar'] != 0:
      active = int(action['hotbar'])
      self.active_item[:] = 0.
      self.active_item[active] = 1.
    item = self.active_item.copy()
    obs, reward, done, info = super().step(action)
    obs['item'] = item
    return obs, reward, done, info
  
  def reset(self):
    obs = super().reset()
    self.active_item[:] = 0.
    self.active_item[0] = 1.
    item = self.active_item.copy()
    obs['item'] = item


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.action_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    assert isinstance(env.action_space[key], gym.spaces.Discrete)
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class RewardObs:

  def __init__(self, env, key='reward'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class ResetObs:

  def __init__(self, env, key='reset'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(0, 1, (), dtype=np.bool)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reset'] = np.array(False, np.bool)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reset'] = np.array(True, np.bool)
    return obs
