"""
Copyright (c) 2020 Jun Zhu
"""
import copy
import random
from collections import namedtuple, deque

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Trainsition", field_names=(
    "state", "action", "reward", "next_state", "done"))


class Memory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size):
        """Initialization.

        :param int buffer_size: maximum size of buffer.
        """
        self._buffer = deque(maxlen=buffer_size)

    def append(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self._buffer.append(
            Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of sequences from memory.

        :param int batch_size: sample batch size.
        """
        experiences = random.sample(self._buffer, k=batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states =np.vstack([
            e.next_state for e in experiences if e is not None])
        dones = np.vstack([
            e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._buffer.__len__()


class DqnAgent:
    """Deep-Q network agent."""

    def __init__(self, model, actions, *,
                 brain_name='BananaBrain',
                 batch_size=16,
                 replay_memory_size=1000,
                 replay_start_size=None,
                 learning_rate=5e-4,
                 discount_factor=0.99,
                 target_network_update_frequency=4,
                 model_file="dqn_checkpoint.pth"):
        """Initialization.

        :param torch.nn.Module model: Q network.
        :param numpy.array actions: available actions.
        :param str brain_name: the brain name of the environment.
        :param int batch_size: mini batch size.
        :param int replay_memory_size: size of the memory buffer.
        :param int replay_start_size: a uniform random policy is run for this
            number of frames before learning starts and the resulting
            experience is used to populate the replay memory.
        :param double learning_rate: learning rate.
        :param double discount_factor: discount factor gamma used in
            Q-learning update.
        :param int target_network_update_frequency: the frequency with which
            the target network updates.
        """
        self._brain_name = brain_name

        # Q-Network
        self._model = model.to(device)
        self._model_target = copy.deepcopy(model).to(device)

        self._actions = actions

        self._optimizer = optim.Adam(self._model.parameters(),
                                     lr=learning_rate)

        self._gamma = discount_factor

        self._target_network_update_frequency = target_network_update_frequency

        self._batch_size = batch_size
        if replay_start_size is None:
            replay_start_size = batch_size * 2
        self._replay_start_size = replay_start_size
        self._memory = Memory(replay_memory_size)

        self._model_file = model_file

    def act(self, state, *, epsilon=0.0):
        """Returns actions for given state as per current policy.

        :param numpy.array state: current state.
        :param float epsilon: for epsilon-greedy action selection. For the
            convenience of the inference stage, the default value is 0.0.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self._model.eval()
        with torch.no_grad():
            action_values = self._model(state)
        self._model.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(self._actions)

    def _learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        :param experiences (Tuple[torch.Variable]): (s, a, r, s', done)
        :param float gamma: discount rate.
        """
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        q_targets_next = self._model_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (self._gamma * q_targets_next * (1 - dones))

        q_expected = self._model(states).gather(1, actions)

        # calculate the element-wise mean-squared error
        loss = F.mse_loss(q_expected, q_targets)

        # clear the gradients of all optimized
        self._optimizer.zero_grad()
        # compute gradient in the backward graph
        loss.backward()
        # perform a single optimization step
        self._optimizer.step()

    def train(self, env, *,
              n_episodes=1000,
              max_steps=100,
              epsilon_decay_rate=0.995,
              window=100,
              target_score=13,
              save_frequency=100,
              output_frequency=10):
        """Train the agent.

        :param gym.Env env: environment.
        :param int n_episodes: number of episodes.
        :param int max_steps: maximum number of time steps.
        :param float epsilon_decay_rate: decay rate of epsilon.
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
        try:
            checkpoint = torch.load(self._model_file)
        except FileNotFoundError:
            checkpoint = None

        eps_initial = 1.0
        eps_final = 0.01

        if checkpoint is None:
            i0 = 0
            eps = eps_initial
            scores = []
        else:
            i0 = checkpoint['epoch']
            eps = checkpoint['epsilon']
            scores = checkpoint['score_history']
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model_target.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded existing model ended at epoch: {i0}")

            avg_score = np.mean(scores[-window:])
            if avg_score > target_score:
                print(f"Score of the current model {avg_score} is already "
                      f"higher than the target score {target_score}!")
                return

        brain_name = self._brain_name
        i = i0
        while i < n_episodes:
            env_info = env.reset(train_mode=True)[brain_name]
            i += 1

            state = env_info.vector_observations[0]
            score = 0
            for j in range(max_steps):
                action = self.act(state, epsilon=eps)
                env_info = env.step(action)[brain_name]
                reward = env_info.rewards[0]
                next_state = env_info.vector_observations[0]
                done = env_info.local_done[0]
                self._memory.append(state, action, reward, next_state, done)
                state = next_state

                if len(self._memory) > self._replay_start_size:
                    self._learn(self._memory.sample(self._batch_size))

                # update target network
                if j % self._target_network_update_frequency == 0:
                    self._model_target.load_state_dict(
                        copy.deepcopy(self._model.state_dict()))

                score += reward
                if done:
                    break

            scores.append(score)

            # update eps
            eps = max(eps_final, epsilon_decay_rate * eps)

            avg_score = np.mean(scores[-window:])
            if i % output_frequency == 0:
                print(f"Epoch: {i:04d}, average score: {avg_score:8.2f}")

            if i % save_frequency == 0:
                self._save_model(i, eps, scores)

            if avg_score >= target_score:
                self._save_model(i, eps, scores)
                break

        if i > i0:
            self._save_model(i, eps, scores)

    def _save_model(self, epoch, eps, scores):
        torch.save({
            'epoch': epoch,
            'epsilon': eps,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'score_history': scores,
        }, self._model_file)
        print(f"Model save in {self._model_file} after {epoch} epochs!")

    def play(self, env):
        env_info = env.reset(train_mode=False)[self._brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = self.act(state)
            env_info = env.step(action)[self._brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break

        print(f"Final score is: {score}")
