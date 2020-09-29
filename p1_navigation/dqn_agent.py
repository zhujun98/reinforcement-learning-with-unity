"""
Copyright (c) 2020 Jun Zhu
"""
import copy
import random

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from agent_base import _AgentBase, Memory
from utilities import copy_nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DqnAgent(_AgentBase):
    """Deep-Q network agent."""

    def __init__(self, model, actions, *,
                 brain_name='BananaBrain',
                 model_file="dqn_checkpoint.pth",
                 replay_memory_size=1000,
                 double_dqn=True):
        """Initialization.

        :param torch.nn.Module model: Q network.
        :param numpy.array actions: available actions.
        :param int replay_memory_size: size of the memory buffer.
        :param bool double_dqn: Use double DQN algorithm to reduce
            overestimation of the value function.
        """
        super().__init__(brain_name, model_file)

        # Q-Network
        self._model = model.to(device)
        self._model_target = copy.deepcopy(model).to(device)

        self._actions = actions

        self._memory = Memory(replay_memory_size)

        self._double_dqn = double_dqn

    def _act(self, state, *, epsilon=0.0):
        """Returns actions for given state as per current policy.

        :param numpy.array state: current state.
        :param float epsilon: for epsilon-greedy action selection. The default
            value is 0.0 which should be used in inference.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self._model.eval()
        with torch.no_grad():
            action_values = self._model(state)
        self._model.train()

        # epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().numpy())
        return random.choice(self._actions)

    def _learn(self, experiences, optimizer, gamma, update=False):
        """Update value parameters using given batch of experience tuples.

        :param (Tuple[torch.Variable]) experiences: (s, a, r, s', done)
        :param Optimizer optimizer: optimizer used for gradient ascend.
        :param float gamma: discount factor.
        :param bool update: True for updating the target network.
        """
        states, actions, rewards, next_states, dones = experiences

        # shape = (batch size, action space size)
        q_next = self._model_target(next_states).detach()
        if self._double_dqn:
            actions_online = torch.argmax(self._model(next_states), dim=-1)
            q_next = q_next.gather(1, actions_online.unsqueeze(-1))
        else:
            q_next = q_next.max(1)[0].unsqueeze(1)

        q_targets = rewards + (gamma * q_next * (1 - dones))
        q_expected = self._model(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update target network
        if update:
            copy_nn(self._model, self._model_target)

    def train(self, env, *,
              n_episodes=1000,
              eps_decay_rate=0.995,
              eps_final=0.01,
              target_network_update_frequency=4,
              gamma=1.0,
              learning_rate=5e-4,
              weight_decay=0,
              batch_size=16,
              replay_start_size=None,
              window=100,
              target_score=13,
              save_frequency=100,
              output_frequency=50):
        """Train the agent.

        :param gym.Env env: environment.
        :param int n_episodes: number of episodes.
        :param float eps_decay_rate: decay rate of epsilon.
        :param float eps_final: final (minimum) epsilon value.
        :param int target_network_update_frequency: the frequency with which
            the target network updates.
        :param double gamma: discount factor gamma used in Q-learning update.
        :param double learning_rate: learning rate.
        :param double weight_decay: L2 penalty.
        :param int batch_size: mini batch size.
        :param int replay_start_size: a uniform random policy is run for this
            number of frames before training starts.
        :param int window: the latest window episodes will be used to evaluate
            the performance of the model.
        :param float target_score: the the average score of the latest window
            episodes is larger than the target score. The problem is considered
            solved.
        :param int save_frequency: the frequency of saving the states of the
            agent.
        :param int output_frequency: the frequency of summarizing the
            training result.
        """
        optimizer = optim.Adam(self._model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)

        try:
            checkpoint = torch.load(self._model_file)
        except FileNotFoundError:
            checkpoint = None

        if checkpoint is None:
            i0 = 0
            scores = []
        else:
            i0 = checkpoint['epoch']
            scores = checkpoint['score_history']
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model_target.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            avg_score = np.mean(scores[-window:])
            print(f"Loaded existing model ended at epoch: {i0} with average"
                  f"score of {avg_score:8.2f}")

            if avg_score > target_score:
                print(f"Score of the current model {avg_score:8.2f} is already "
                      f"higher than the target score {target_score}!")
                return scores

        if replay_start_size is None:
            replay_start_size = batch_size * 2

        brain_name = self._brain_name
        i = i0
        eps = max(eps_final, eps_decay_rate ** i)
        while i < n_episodes:
            i += 1

            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            i_step = 0  # each episode has 300 steps
            while True:
                i_step += 1

                action = self._act(state, epsilon=eps)
                env_info = env.step(action)[brain_name]
                reward = env_info.rewards[0]
                score += reward
                next_state = env_info.vector_observations[0]
                done = env_info.local_done[0]
                self._memory.append(state, action, reward, next_state, done)
                state = next_state

                if len(self._memory) > replay_start_size:
                    self._learn(self._memory.sample(batch_size, device=device),
                                optimizer,
                                gamma,
                                i_step % target_network_update_frequency == 0)

                if done:
                    break

            scores.append(score)

            # update eps
            eps = max(eps_final, eps_decay_rate * eps)

            avg_score = np.mean(scores[-window:])

            if avg_score >= target_score:
                print(f"Epoch: {i:04d}, average score: {avg_score:8.2f}")
                self._save_model(i, optimizer, scores)
                break

            if i % output_frequency == 0:
                print(f"Epoch: {i:04d}, average score: {avg_score:8.2f}")

            if i % save_frequency == 0:
                self._save_model(i, optimizer, scores)

        return scores

    def _save_model(self, epoch, optimizer, scores):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score_history': scores,
        }, self._model_file)
        print(f"Model save in {self._model_file} after {epoch} epochs!")
