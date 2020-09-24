"""
Copyright (c) 2020 Jun Zhu
"""
import random

import numpy as np
import torch

from collections import namedtuple, deque

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

    def sample(self, batch_size, device=None):
        """Randomly sample a batch of sequences from memory.

        :param int batch_size: sample batch size.
        """
        experiences = random.sample(self._buffer, k=batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([
            e.next_state for e in experiences if e is not None])
        dones = np.vstack([
            e.done for e in experiences if e is not None]).astype(np.uint8)

        if device is not None:
            states = torch.from_numpy(states).float().to(device)
            actions = torch.from_numpy(actions).float().to(device)
            rewards = torch.from_numpy(rewards).float().to(device)
            next_states = torch.from_numpy(next_states).float().to(device)
            dones = torch.from_numpy(dones).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._buffer.__len__()


class _AgentBase:
    """Base class for the agent."""
    def __init__(self, brain_name, model_file):
        """Initialization.

        :param str brain_name: the brain name of the environment.
        :param str model_file: file to save the trained model.
        """
        self._brain_name = brain_name
        self._model_file = model_file

        self._n_agents = 1

    def _act(self, *args, **kwargs):
        raise NotImplementedError

    def play(self, env):
        """Play the environment once."""
        env_info = env.reset(train_mode=False)[self._brain_name]
        states = env_info.vector_observations
        scores = [0] * self._n_agents
        while True:
            actions = self._act(states)
            env_info = env.step(actions)[self._brain_name]
            next_states = env_info.vector_observations
            states = next_states
            rewards = env_info.rewards
            for i_a in range(self._n_agents):
                scores[i_a] += rewards[i_a]

            if env_info.local_done[0]:
                break

        return max(scores)
