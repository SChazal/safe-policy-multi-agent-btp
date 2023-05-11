import copy
from datetime import datetime
from functional import seq
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam

from safe_explorer.core.config import Config
from safe_explorer.core.replay_buffer import ReplayBuffer
from safe_explorer.core.tensorboard import TensorBoard
from safe_explorer.utils.list import for_each, select_with_predicate

class DDPG_multi:
    def __init__(self,
                 env,
                 actor_1,
                 critic_1,
                 actor_2,
                 critic_2,
                 action_modifier_1=None,
                 action_modifier_2=None):
        self._env = env
        self._actor_1 = actor_1
        self._critic_1 = critic_1
        self._actor_2 = actor_2
        self._critic_2 = critic_2
        self._action_modifier_1 = action_modifier_1
        self._action_modifier_2 = action_modifier_2

        self._config = Config.get().ddpg.trainer

        self._initialize_target_networks()
        self._initialize_optimizers()

        self._models_1 = {
            'actor': self._actor_1,
            'critic': self._critic_1,
            'target_actor': self._target_actor_1,
            'target_critic': self._target_critic_1
        }
        
        self._models_2 = {
            'actor': self._actor_2,
            'critic': self._critic_2,
            'target_actor': self._target_actor_2,
            'target_critic': self._target_critic_2
        }

        self._replay_buffer_1 = ReplayBuffer(self._config.replay_buffer_size)
        self._replay_buffer_2 = ReplayBuffer(self._config.replay_buffer_size)

        # Tensorboard writer
        self._writer = TensorBoard.get_writer()
        self._train_global_step = 0
        self._eval_global_step = 0

        if self._config.use_gpu:
            self._cuda()

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        if self._config.use_gpu:
            tensor = tensor.cuda()
        return tensor

    def _initialize_target_networks(self):
        self._target_actor_1 = copy.deepcopy(self._actor_1)
        self._target_critic_1 = copy.deepcopy(self._critic_1)

        self._target_actor_2 = copy.deepcopy(self._actor_2)
        self._target_critic_2 = copy.deepcopy(self._critic_2)
    
    def _initialize_optimizers(self):
        self._actor_1_optimizer = Adam(self._actor_1.parameters(), lr=self._config.actor_lr)
        self._critic_1_optimizer = Adam(self._critic_1.parameters(), lr=self._config.critic_lr)

        self._actor_2_optimizer = Adam(self._actor_2.parameters(), lr=self._config.actor_lr)
        self._critic_2_optimizer = Adam(self._critic_2.parameters(), lr=self._config.critic_lr)
    
    def _eval_mode(self):
        for_each(lambda x: x.eval(), self._models_1.values())
        for_each(lambda x: x.eval(), self._models_2.values())

    def _train_mode(self):
        for_each(lambda x: x.train(), self._models_1.values())
        for_each(lambda x: x.train(), self._models_2.values())

    def _cuda(self):
        for_each(lambda x: x.cuda(), self._models_1.values())
        for_each(lambda x: x.cuda(), self._models_2.values())

    def _get_action_1(self, observation, c, is_training=True):
        # Action + random gaussian noise (as recommended in spinning up)
        action = self._actor_1(self._as_tensor(self._flatten_dict(observation)))

        if self._config.use_gpu:
            action = action.detach().cpu() #<--------

        if is_training:
            action += self._config.action_noise_range * torch.randn(self._env.action_space.shape)

        action = action.data.numpy()

        if self._action_modifier_1:
            action = self._action_modifier_1(observation, action, c) 

        return action
    
    def _get_action_2(self, observation, c, is_training=True):
        # Action + random gaussian noise (as recommended in spinning up)
        action = self._actor_2(self._as_tensor(self._flatten_dict(observation)))

        if self._config.use_gpu:
            action = action.detach().cpu() #<--------

        if is_training:
            action += self._config.action_noise_range * torch.randn(self._env.action_space.shape)

        action = action.data.numpy()

        if self._action_modifier_2:
            action = self._action_modifier_2(observation, action, c) 

        return action

    def _get_target_1(self, batch):
        # For each observation in batch:
        # target = r + discount_factor * (1 - done) * max_a Q_tar(s, a)
        # a => actions of actor on current observations
        # max_a Q_tar(s, a) = output of critic
        observation_next = self._as_tensor(batch["observation_next"])
        reward = self._as_tensor(batch["reward"]).reshape(-1, 1)
        done = self._as_tensor(batch["done"]).reshape(-1, 1)

        action = self._target_actor_1(observation_next).reshape(-1, *self._env.action_space.shape)

        q = self._target_critic_1(observation_next, action)

        return reward  + self._config.discount_factor * (1 - done) * q
    
    def _get_target_2(self, batch):
        # For each observation in batch:
        # target = r + discount_factor * (1 - done) * max_a Q_tar(s, a)
        # a => actions of actor on current observations
        # max_a Q_tar(s, a) = output of critic
        observation_next = self._as_tensor(batch["observation_next"])
        reward = self._as_tensor(batch["reward"]).reshape(-1, 1)
        done = self._as_tensor(batch["done"]).reshape(-1, 1)

        action = self._target_actor_2(observation_next).reshape(-1, *self._env.action_space.shape)

        q = self._target_critic_2(observation_next, action)

        return reward  + self._config.discount_factor * (1 - done) * q

    def _flatten_dict(self, inp):
        if type(inp) == dict:
            inp = np.concatenate(list(inp.values()))
        return inp

    def _update_targets(self, target, main):
        for target_param, main_param in zip(target.parameters(), main.parameters()):
            target_param.data.copy_(self._config.polyak * target_param.data + \
                                    (1 - self._config.polyak) * main_param.data)

    def _update_batch_1(self):
        batch = self._replay_buffer_1.sample(self._config.batch_size)
        # Only pick steps in which action was non-zero
        # When a constraint is violated, the safety layer makes action 0 in
        # direction of violating constraint
        # valid_action_mask = np.sum(batch["action"], axis=1) > 0
        # batch = {k: v[valid_action_mask] for k, v in batch.items()}

        # Update critic
        self._critic_1_optimizer.zero_grad()
        q_target = self._get_target_1(batch)
        q_predicted = self._critic_1(self._as_tensor(batch["observation"]),
                                   self._as_tensor(batch["action"]))
        # critic_loss = torch.mean((q_predicted.detach() - q_target) ** 2)
        # Seems to work better
        critic_loss = F.smooth_l1_loss(q_predicted, q_target)

        critic_loss.backward()
        self._critic_1_optimizer.step()

        # Update actor
        self._actor_1_optimizer.zero_grad()
        # Find loss with updated critic
        new_action = self._actor_1(self._as_tensor(batch["observation"])).reshape(-1, *self._env.action_space.shape)
        actor_loss = -torch.mean(self._critic_1(self._as_tensor(batch["observation"]), new_action))
        actor_loss.backward()
        self._actor_1_optimizer.step()

        # Update targets networks
        self._update_targets(self._target_actor_1, self._actor_1)
        self._update_targets(self._target_critic_1, self._critic_1)

        # Log to tensorboard
        self._writer.add_scalar("critic_1 loss", critic_loss.item(), self._train_global_step)
        self._writer.add_scalar("actor_1 loss", actor_loss.item(), self._train_global_step)
        # (seq(self._models.items())
        #             .flat_map(lambda x: [(x[0], y) for y in x[1].named_parameters()]) # (model_name, (param_name, param_data))
        #             .map(lambda x: (f"{x[0]}_{x[1][0]}", x[1][1]))
        #             .for_each(lambda x: self._writer.add_histogram(x[0], x[1].data.numpy(), self._train_global_step)))
        # self._train_global_step +=1 #update global step after second agents update
    
    def _update_batch_2(self):
        batch = self._replay_buffer_2.sample(self._config.batch_size)
        # Only pick steps in which action was non-zero
        # When a constraint is violated, the safety layer makes action 0 in
        # direction of violating constraint
        # valid_action_mask = np.sum(batch["action"], axis=1) > 0
        # batch = {k: v[valid_action_mask] for k, v in batch.items()}

        # Update critic
        self._critic_2_optimizer.zero_grad()
        q_target = self._get_target_2(batch)
        q_predicted = self._critic_2(self._as_tensor(batch["observation"]),
                                   self._as_tensor(batch["action"]))
        # critic_loss = torch.mean((q_predicted.detach() - q_target) ** 2)
        # Seems to work better
        critic_loss = F.smooth_l1_loss(q_predicted, q_target)

        critic_loss.backward()
        self._critic_2_optimizer.step()

        # Update actor
        self._actor_2_optimizer.zero_grad()
        # Find loss with updated critic
        new_action = self._actor_2(self._as_tensor(batch["observation"])).reshape(-1, *self._env.action_space.shape)
        actor_loss = -torch.mean(self._critic_2(self._as_tensor(batch["observation"]), new_action))
        actor_loss.backward()
        self._actor_2_optimizer.step()

        # Update targets networks
        self._update_targets(self._target_actor_2, self._actor_2)
        self._update_targets(self._target_critic_2, self._critic_2)

        # Log to tensorboard
        self._writer.add_scalar("critic_2 loss", critic_loss.item(), self._train_global_step)
        self._writer.add_scalar("actor_2 loss", actor_loss.item(), self._train_global_step)
        # (seq(self._models.items())
        #             .flat_map(lambda x: [(x[0], y) for y in x[1].named_parameters()]) # (model_name, (param_name, param_data))
        #             .map(lambda x: (f"{x[0]}_{x[1][0]}", x[1][1]))
        #             .for_each(lambda x: self._writer.add_histogram(x[0], x[1].data.numpy(), self._train_global_step)))
        self._train_global_step +=1

    def _update(self, episode_length):
        # Update model #episode_length times
        for_each(lambda x: self._update_batch_1(),
                 range(min(episode_length, self._config.max_updates_per_episode)))
        for_each(lambda x: self._update_batch_2(),
                 range(min(episode_length, self._config.max_updates_per_episode)))

    def evaluate(self):
        episode_rewards_1 = []
        episode_rewards_2 = []
        episode_lengths = []
        episode_actions_1 = []
        episode_actions_2 = []

        observation_1, observation_2 = self._env.reset()
        c_1, c_2 = self._env.get_constraint_values()

        episode_reward_1 = 0
        episode_reward_2 = 0
        episode_length = 0
        episode_action_1 = 0
        episode_action_2 = 0

        self._eval_mode()

        for step in range(self._config.evaluation_steps):
            action_1 = self._get_action_1(observation_1, c_1, is_training=False)
            episode_action_1 += np.absolute(action_1)
            action_2 = self._get_action_2(observation_2, c_2, is_training=False)
            episode_action_2 += np.absolute(action_2)
            observation_1_next, reward_1, observation_2_next, reward_2, done, _ = self._env.step(action_1, action_2)
            c_1, c_2 = self._env.get_constraint_values()
            episode_reward_1 += reward_1
            episode_reward_2 += reward_2
            episode_length += 1
            
            if done or (episode_length == self._config.max_episode_length):
                episode_rewards_1.append(episode_reward_1)
                episode_rewards_2.append(episode_reward_2)
                episode_lengths.append(episode_length)
                episode_actions_1.append(episode_action_1 / episode_length)
                episode_actions_2.append(episode_action_2 / episode_length)

                observation_1, observation_2 = self._env.reset()
                c_1, c_2 = self._env.get_constraint_values()
                episode_reward_1 = 0
                episode_reward_2 = 0
                episode_length = 0
                episode_action_1 = 0
                episode_action_2 = 0

        mean_episode_reward_1 = np.mean(episode_rewards_1)
        mean_episode_reward_2 = np.mean(episode_rewards_2)
        mean_episode_length = np.mean(episode_lengths)

        self._writer.add_scalar("eval mean episode reward_1", mean_episode_reward_1, self._eval_global_step)
        self._writer.add_scalar("eval mean episode reward_2", mean_episode_reward_2, self._eval_global_step)
        self._writer.add_scalar("eval mean action_1 magnitude", np.mean(episode_actions_1), self._eval_global_step)
        self._writer.add_scalar("eval mean action_2 magnitude", np.mean(episode_actions_2), self._eval_global_step)
        self._writer.add_scalar("eval mean episode length", mean_episode_length, self._eval_global_step)
        self._eval_global_step += 1

        self._train_mode()

        print("Validation completed:\n"
              f"Number of episodes: {len(episode_actions_1)}\n"
              f"Average episode length: {mean_episode_length}\n"
              f"Average reward_1: {mean_episode_reward_1}\n"
              f"Average reward_2: {mean_episode_reward_2}\n"
              f"Average action_1 magnitude: {np.mean(episode_actions_1)}\n"
              f"Average action_2 magnitude: {np.mean(episode_actions_2)}")

    def train(self):
        
        start_time = time.time()

        print("==========================================================")
        print("Initializing DDPG training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        observation_1, observation_2 = self._env.reset()
        c_1, c_2 = self._env.get_constraint_values()

        episode_reward_1 = 0
        episode_reward_2 = 0
        episode_length = 0

        number_of_steps = self._config.steps_per_epoch * self._config.epochs

        for step in range(number_of_steps):
            # Randomly sample episode_ for some initial steps
            action_1 = self._env.action_space.sample() if step < self._config.start_steps \
                     else self._get_action_1(observation_1, c_1)

            action_2 = self._env.action_space.sample() if step < self._config.start_steps \
                     else self._get_action_2(observation_2, c_2)

            observation_1_next, reward_1, observation_2_next, reward_2, done, _ = self._env.step(action_1, action_2)
            episode_reward_1 += reward_1
            episode_reward_2 += reward_2
            episode_length += 1

            self._replay_buffer_1.add({
                "observation": self._flatten_dict(observation_1),
                "action": action_1,
                "reward": np.asarray(reward_1) * self._config.reward_scale,
                "observation_next": self._flatten_dict(observation_1_next),
                "done": np.asarray(done),
            })

            self._replay_buffer_2.add({
                "observation": self._flatten_dict(observation_2),
                "action": action_2,
                "reward": np.asarray(reward_2) * self._config.reward_scale,
                "observation_next": self._flatten_dict(observation_2_next),
                "done": np.asarray(done),
            })

            observation_1 = observation_1_next
            observation_2 = observation_2_next

            c_1, c_2 = self._env.get_constraint_values()

            # Make all updates at the end of the episode
            if done or (episode_length == self._config.max_episode_length):
                if step >= self._config.min_buffer_fill:
                    self._update(episode_length)
                    self._writer.add_scalar("episode length", episode_length, self._train_global_step)
                    self._writer.add_scalar("episode reward_1", episode_reward_1, self._train_global_step)
                    self._writer.add_scalar("episode reward_2", episode_reward_2, self._train_global_step)
                # Reset episode
                observation_1, observation_2 = self._env.reset()
                c_1, c_2 = self._env.get_constraint_values()
                episode_reward_1 = 0
                episode_reward_2 = 0
                episode_length = 0
                

            # Check if the epoch is over
            if step != 0 and step % self._config.steps_per_epoch == 0: 
                print(f"Finished epoch {step / self._config.steps_per_epoch}. Running validation ...")
                self.evaluate()
                print("----------------------------------------------------------")
        
        self._writer.close()
        print("==========================================================")
        print(f"Finished DDPG training. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")