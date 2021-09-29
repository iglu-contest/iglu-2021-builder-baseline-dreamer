from sys import prefix
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import elements
import common
import expl


class Agent(common.Module):

  def __init__(self, config, logger, actspce, step, dataset):
    self.config = config
    self._logger = logger
    self._action_space = actspce
    self._num_act = actspce.n if hasattr(actspce, 'n') else actspce.shape[0]
    self._should_expl = elements.Until(int(
        config.expl_until / config.action_repeat))
    self._counter = step
    with tf.device('cpu:0'):
      self.step = tf.Variable(int(self._counter), tf.int64)
    self._dataset = dataset
    self.wm = WorldModel(self.step, config)
    self._task_behavior = ActorCritic(config, self.step, self._num_act)
    self.reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(actspce),
        plan2explore=lambda: expl.Plan2Explore(
            config, self.wm, self._num_act, self.step, self.reward),
        model_loss=lambda: expl.ModelLoss(
            config, self.wm, self._num_act, self.step, self.reward),
    )[config.expl_behavior]()
    # Train step to initialize variables including optimizer statistics.
    data = next(self._dataset)
    self.train(data)
    pass

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    print('calling policy')
    tf.py_function(lambda: self.step.assign(
        int(self._counter), read_value=False), [], [])
    if state is None:
      # todo: rewrite to make this anywhere else
      inventory = tf.constant([[20, 20, 20, 20, 20, 20]], dtype=tf.float32)
      grid = tf.zeros((1, 9 * 11 * 11), dtype=tf.int32)
      grid = tf.gather(tf.eye(7), grid)
      grid = tf.reshape(grid, tf.concat([tf.shape(grid)[:1], tf.constant([-1])], 0))
      pos = tf.constant([[0.5, 0., 0.5, 0., -90.]], dtype=tf.float32)
      pre_state = tf.concat([pos, inventory, grid], 1)
      latent = self.wm.rssm.init_from_vec(pre_state)
      # latent = self.wm.rssm.initial(len(obs['image']))
      action = tf.zeros((len(obs['image']), self._num_act))
      state = latent, action
    elif obs['reset'].any():
      state = tf.nest.map_structure(lambda x: x * common.pad_dims(
          1.0 - tf.cast(obs['reset'], x.dtype), len(x.shape)), state)
    latent, action = state
    data = self.wm.preprocess(obs)
    embed = self.wm.encoder(data)
    vec_embed = self.wm.inventory_compass_encoder(data['inventory_compass'])
    if self.config.encode_grid:
      grid_embed = self.wm.preprocess_state(data, prefix='')
      embed = tf.concat([embed, vec_embed, grid_embed], -1)
    else:
      embed = tf.concat([embed, vec_embed], -1)
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, sample)
    feat = self.wm.rssm.get_feat(latent)
    if mode == 'eval':
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
    elif self._should_expl(self.step):
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
    noise = {'train': self.config.expl_noise, 'eval': self.config.eval_noise}
    action = common.action_noise(action, noise[mode], self._action_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train(self, data, state=None):
    print('calling train agent')
    metrics = {}
    state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)
    start = outputs['post']
    if self.config.pred_discount:  # Last step could be terminal.
      start = tf.nest.map_structure(lambda x: x[:, :-1], start)
    reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
    metrics.update(self._task_behavior.train(self.wm, start, reward))
    if self.config.expl_behavior != 'greedy':
      if self.config.pred_discount:
        data = tf.nest.map_structure(lambda x: x[:, :-1], data)
        outputs = tf.nest.map_structure(lambda x: x[:, :-1], outputs)
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    return state, metrics

  @tf.function
  def report(self, data):
    if not self.config.disable_decoder:
      return {'openl': self.wm.video_pred(data)}
    return {}


class WorldModel(common.Module):

  def __init__(self, step, config):
    self.step = step
    self.config = config
    self.rssm = common.RSSM(**config.rssm)
    self.heads = {}
    shape = config.image_size + (config.img_channels,)
    self.encoder = common.ConvEncoder(**config.encoder)
    self.inventory_compass_encoder = common.EncoderMLP(**config.vec_encoder)
    if not config.disable_decoder:
      self.heads['image'] = common.ConvDecoder(shape, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    self.heads['grid'] = common.MLP(**config.grid_head)
    self.heads['position'] = common.MLP(**config.pos_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)

  def train(self, data, state=None):
    print('calling train wm')
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.rssm, self.inventory_compass_encoder, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def preprocess_state(self, data, prefix='init_', dtype=tf.float32):
    grid = tf.gather(tf.eye(7), data[f'{prefix}grid'])
    grid = tf.reshape(grid, tf.concat([tf.shape(grid)[:-2], tf.constant([-1])], 0))
    grid = tf.cast(grid, dtype)
    position = tf.cast(data[f'{prefix}position'], dtype)
    inventory = tf.cast(data[f'{prefix}inventory'], dtype)
    state = tf.concat([position, inventory, grid], -1)
    dtype = prec.global_policy().compute_dtype
    state = tf.cast(state, dtype)
    return state

  def loss(self, data, state=None):
    if state is None:
      state = self.preprocess_state(data)
    print('calling wm loss')
    data = self.preprocess(data)
    embed = self.encoder(data)
    vec_embed = self.inventory_compass_encoder(data['inventory_compass'])
    if self.config.encode_grid:
      grid_embed = self.preprocess_state(data, prefix='')
      embed = tf.concat([embed, vec_embed, grid_embed], 2)
    else:
      embed = tf.concat([embed, vec_embed], 2)
    post, prior = self.rssm.observe(embed, data['action'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
    assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      like = tf.cast(head(inp).log_prob(data[name]), tf.float32)
      likes[name] = like
      if name == 'grid':
        loss = -like.sum(-1).mean()
      else:
        loss = -like.mean()
      losses[name] = loss
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    return model_loss, post, outs, metrics

  def imagine(self, policy, start, horizon):
    print('calling wm imagine')
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    def step(prev, _):
      state, _, _ = prev
      feat = self.rssm.get_feat(state)
      action = policy(tf.stop_gradient(feat)).sample()
      succ = self.rssm.img_step(state, action)
      return succ, feat, action
    feat = 0 * self.rssm.get_feat(start)
    action = policy(feat).mode()
    succs, feats, actions = common.static_scan(
        step, tf.range(horizon), (start, feat, action))
    states = {k: tf.concat([
        start[k][None], v[:-1]], 0) for k, v in succs.items()}
    if 'discount' in self.heads:
      discount = self.heads['discount'](feats).mean()
    else:
      discount = self.config.discount * tf.ones_like(feats[..., 0])
    return feats, states, actions, discount

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    obs['compass'] = tf.cast(obs['compass'], dtype)
    obs['inventory'] = tf.cast(obs['inventory'], dtype)
    compass = tf.expand_dims(obs['compass'], len(tf.shape(obs['compass'])))
    obs['inventory_compass'] = tf.concat([obs['inventory'], compass], len(obs['inventory'].shape) - 1)
    obs['inventory_compass'] = tf.cast(obs['inventory_compass'], dtype)
    obs['reward'] = getattr(tf, self.config.clip_rewards)(obs['reward'])
    if 'discount' in obs:
      obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data):
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    embed = self.encoder(data)
    vec_embed = self.inventory_compass_encoder(data['inventory_compass'])
    if self.config.encode_grid:
      grid_embed = self.preprocess_state(data, prefix='')
      embed = tf.concat([embed, vec_embed, grid_embed], 2)
    else:
      embed = tf.concat([embed, vec_embed], 2)
    state = self.preprocess_state(data)
    states, _ = self.rssm.observe(embed[:6, :5], data['action'][:6, :5], state[:6])
    recon = self.heads['image'](
        self.rssm.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    openl = self.heads['image'](self.rssm.get_feat(prior)).mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    _, _, h, w, _ = model.shape
    video = tf.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    video = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    return tf.concat(tf.split(video, C // 3, 3), 1)

class ActorCritic(common.Module):

  def __init__(self, config, step, num_actions):
    self.config = config
    self.step = step
    self.num_actions = num_actions
    self.actor = common.MLP(num_actions, **config.actor)
    self.critic = common.MLP([], **config.critic)
    if config.slow_target:
      self._target_critic = common.MLP([], **config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', **config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **config.critic_opt)

  def train(self, world_model, start, reward_fn):
    print('calling ac train')
    metrics = {}
    hor = self.config.imag_horizon
    with tf.GradientTape() as actor_tape:
      feat, state, action, disc = world_model.imagine(self.actor, start, hor)
      reward = reward_fn(feat, state, action)
      target, weight, mets1 = self.target(feat, action, reward, disc)
      actor_loss, mets2 = self.actor_loss(feat, action, target, weight)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets3 = self.critic_loss(feat, action, target, weight)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, feat, action, target, weight):
    print('calling a loss')
    metrics = {}
    policy = self.actor(tf.stop_gradient(feat))
    if self.config.actor_grad == 'dynamics':
      objective = target
    elif self.config.actor_grad == 'reinforce':
      baseline = self.critic(feat[:-1]).mode()
      advantage = tf.stop_gradient(target - baseline)
      objective = policy.log_prob(action)[:-1] * advantage
    elif self.config.actor_grad == 'both':
      baseline = self.critic(feat[:-1]).mode()
      advantage = tf.stop_gradient(target - baseline)
      objective = policy.log_prob(action)[:-1] * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.step)
      objective = mix * target + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.step)
    objective += ent_scale * ent[:-1]
    actor_loss = -(weight[:-1] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, feat, action, target, weight):
    print('calling c loss')
    dist = self.critic(feat)[:-1]
    target = tf.stop_gradient(target)
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, feat, action, reward, disc):
    reward = tf.cast(reward, tf.float32)
    disc = tf.cast(disc, tf.float32)
    value = self._target_critic(feat).mode()
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1], lambda_=self.config.discount_lambda, axis=0)
    weight = tf.stop_gradient(tf.math.cumprod(tf.concat(
        [tf.ones_like(disc[:1]), disc[:-1]], 0), 0))
    metrics = {}
    metrics['reward_mean'] = reward.mean()
    metrics['reward_std'] = reward.std()
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, weight, metrics

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)
