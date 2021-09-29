import datetime
import io
import pathlib
import os
import uuid

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec


class Replay:

  def __init__(self, directory, limit=None, rescan=1, cache=None):
    directory.mkdir(parents=True, exist_ok=True)
    self._directory = directory
    self._limit = limit
    self._rescan = rescan
    self._cache = cache
    self._step = sum(int(
        str(n).split('-')[-1][:-4]) - 1 for n in directory.glob('*.npz'))
    self._episodes = load_episodes(directory, limit)

  @property
  def total_steps(self):
    return self._step

  @property
  def num_episodes(self):
    return len(self._episodes)

  @property
  def num_transitions(self):
    return sum(int(
        str(n).split('-')[-1][:-4]) - 1 for n in self._directory.glob('*.npz'))

  def add(self, episode):
    length = self._length(episode)
    self._step += length
    if self._limit:
      total = 0
      for key, ep in reversed(sorted(
          self._episodes.items(), key=lambda x: x[0])):
        if total <= self._limit - length:
          total += self._length(ep)
        else:
          del self._episodes[key]
    filename = save_episodes(self._directory, [episode])[0]
    # self._episodes[str(filename)] = episode
    return filename

  def preproc(self, config):
    def inner(obs):
      dtype = prec.global_policy().compute_dtype
      obs = obs.copy()
      # second preprocessing for multitask ???
      obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
      obs['reward'] = getattr(tf, config.clip_rewards)(obs['reward'])
      if 'discount' in obs:
        obs['discount'] *= config.discount
      return obs
    return inner

  def dataset(self, batch, length, oversample_ends, config):
    if len(self._episodes) == 0:
      self._episodes = load_episodes(self._directory, limit=10)
    example = self._episodes[next(iter(self._episodes.keys()))]
    types = {k: v.dtype for k, v in example.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
    for name in ['position', 'grid', 'inventory']:
      types[f'init_{name}'] = types[name]
      shapes[f'init_{name}'] = (None,) + example[name].shape[2:]
    generator = lambda: sample_episodes(
        self._directory, length, 
        oversample_ends, rescan=self._rescan, cache=self._cache
    )
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.prefetch(10)
    return dataset

  def _length(self, episode):
    return len(episode['reward']) - 1


def save_episodes(directory, episodes):
  directory = pathlib.Path(directory).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  filenames = []
  for episode in episodes:
    identifier = str(uuid.uuid4().hex)
    length = len(episode['reward']) - 1
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
      np.savez_compressed(f1, **episode)
      f1.seek(0)
      with filename.open('wb') as f2:
        f2.write(f1.read())
    filenames.append(filename)
  return filenames


def sample_episodes(directory=None, length=None, balance=False, rescan=100, cache=None, seed=0):
  random = np.random.RandomState(seed)
  while True:
    if cache:
      _, episode = next(iter(load_episodes_lazy(directory, limit=10)))
      ep_len = len(episode['reward']) - 1
      episodes = {}
      for key, val in load_episodes(directory, limit=int(cache or 0) * ep_len, random=True).items():
        episodes[key] = val
    for _ in range(rescan):
      episode = random.choice(list(episodes.values()))
      if length:
        total = len(next(iter(episode.values())))
        available = total - length
        if available < 1:
          continue
        if balance:
          index = min(random.randint(0, total), available)
        else:
          index = int(random.randint(0, available + 1))
        init = {}
        init['init_position'] = episode['position'][max(index - 1, 0)]
        init['init_grid'] = episode['grid'][max(index - 1, 0)]
        init['init_inventory'] = episode['inventory'][max(index - 1, 0)]
        episode = {k: v[index: index + length] for k, v in episode.items()}
        episode.update(init)
        # add initial state to the batch
        # print({k: v.shape for k, v in episode.items()})
        
      yield episode


def load_episodes_lazy(directory, limit=None, random=False):
  directory = pathlib.Path(directory).expanduser()
  total = 0
  if random:
    paths = list(directory.glob('*.npz'))
    paths = [paths[i] for i in np.random.permutation(len(paths))]
  else:
    paths = reversed(sorted(directory.glob('*.npz')))
  for filename in paths:
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode: {e}')
      continue
    yield str(filename), episode
    total += len(episode['reward']) - 1
    if limit and total >= limit:
      break

def load_episodes(directory, limit=None, random=False):
  directory = pathlib.Path(directory).expanduser()
  episodes = {}
  total = 0
  if random:
    paths = list(directory.glob('*.npz'))
    paths = [paths[i] for i in np.random.permutation(len(paths))]
  else:
    paths = reversed(sorted(directory.glob('*.npz')))
  for filename in paths:
    try:
      with filename.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    except Exception as e:
      print(f'Could not load episode: {e}')
      continue
    episodes[str(filename)] = episode
    total += len(episode['reward']) - 1
    if limit and total >= limit:
      break
  return episodes
