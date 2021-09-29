import os
from collections import defaultdict
import yaml
import gym
import numpy as np

NB_EPISODES = 3
MAX_EPISODE_STEPS = 1000
VISUAL = True

def check_action(action, action_space):
    if not isinstance(action, dict):
        raise ValueError('action should be a dict')
    for k in action:
        if k not in action_space.spaces:
            raise ValueError('unexpected action key: {}'.format(k))

def _play_game(agent_class, env_spec, env=None):
    """
    Args:
        agent_class:
        env_specs (path to file or dict): path to yaml or dict with environment.
        env: 
    """
    # To make things faster, set this to '0'
    os.environ['IGLU_DISABLE_FAKE_RESET'] = '0'
    import iglu
    from iglu.tasks import TaskSet, Task, CustomTasks

    stats = defaultdict(lambda: defaultdict(list))
    if isinstance(env_spec, str) and os.path.exists(env_spec):
        with open(env_spec, 'r') as f:
            data = yaml.safe_load(f)
        env_spec = data
    if env is None:
        requested_action_space = env_spec['action_space']
        name = f'IGLUSilentBuilder{"Visual" if VISUAL else ""}-v0'
        print(f'Running {name} using {requested_action_space} action space...')
        env = gym.make(
            name, 
            max_steps=MAX_EPISODE_STEPS,
            action_space=requested_action_space
        )

    agent = agent_class(action_space=env.action_space)

    # here we set the current structure as the task of the current environment 
    custom_grid = np.zeros((9, 11, 11)) # (y, x, z)
    custom_grid[:3, 5, 5] = 1 # blue color
    custom_grid[0, 4, 5] = 1 # blue color
    custom_grid[0, 3, 5] = 1 # blue color
    env.update_taskset(CustomTasks([
        ('<Architect> Please, build a stack of three red blocks somewhere.\n'
        '<Builder> Sure.',
        custom_grid)
    ]))
    task = '<fake_task_id>'

    for episode in range(NB_EPISODES):
        obs = env.reset()
        target_grid_size = len(env.tasks.current.target_grid.nonzero()[0])
        done = False
        reward = 0
        total_reward = 0
        info = {}
        if VISUAL: 
            # remove the grid key which was needed only for reward computation
            del obs['grid']
        else:
            # expose the target grid after reset
            info['target_grid'] = env.tasks.current.target_grid.copy()

        maximal_intersection = 0

        while not done:
            action = agent.act(obs, reward, done, info)
            check_action(action, env.action_space)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            maximal_intersection = max(env.task.task_monitor.max_int, maximal_intersection)

            # just for sanity check
            if maximal_intersection > target_grid_size:
                raise ValueError('intersetion cannot be bigger than a part of it.'
                                'Probably, the task inside the env is wrong')

        # Let the agent know the game is done.
        agent.act(obs, reward, done, info)
        stats[task]['reward'].append(total_reward)
        sr = float(maximal_intersection == target_grid_size)
        stats[task]['success_rate'].append(sr)
        cr = maximal_intersection / target_grid_size
        stats[task]['completion_rate'].append(cr)
        print(f'Episode {episode}/{NB_EPISODES} of task {task}: '
                f'reward={total_reward}; succ_rate={sr}; compl_rate={cr}')
    stats[task]['action_space'] = requested_action_space

    env.close()

    return stats

if __name__ == '__main__':
    from custom_agent import CustomAgent
    _play_game(CustomAgent, 'metadata')