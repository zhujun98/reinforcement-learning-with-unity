"""
Copyright (c) 2020 Jun Zhu
"""
import random

import numpy as np

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

    def sample(self, batch_size):
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

    def _act(self, state):
        raise NotImplementedError

    def play(self, env):
        """Play the environment once."""
        env_info = env.reset(train_mode=False)[self._brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = self._act(state)
            env_info = env.step(action)[self._brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break

        return score
