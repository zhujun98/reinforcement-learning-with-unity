"""
Copyright (c) 2020 Jun Zhu
"""
import numpy as np


def check_environment(env):
    """Check environment.

    :param UnityEnvironment env: unity environment.
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    print("Brain name: ", brain_name)
    print("Brain: ", brain)

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    return brain_name, state_size, action_size


def play(env, brain_name=None, agent=None, continuous=False):
    score = 0
    if agent is None:
        # play randomly
        brain = env.brains[brain_name]
        action_size = brain.vector_action_space_size
        env.reset(train_mode=False)
        while True:
            if continuous:
                action = np.clip(np.random.randn(1, action_size), -1, 1)
            else:
                action = np.random.randint(action_size)

            env_info = env.step(action)[brain_name]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            if done:
                break
    else:
        # use a trained agent
        agent.play(env)

    print("Score: {}".format(score))


def plot_score_history(ax, scores, target_score):
    x = np.arange(len(scores)) + 1
    ax.plot(x, scores, alpha=180, label='score history')
    ax.plot(x[99:], np.convolve(scores, np.ones(100) / 100, 'valid'),
            label='score history (moving averaged)')
    ax.plot(x, target_score * np.ones_like(x), '--', label='reward (target)')
    ax.set_xlabel("Episode", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.legend()


class OUProcess:
    """Ornstein-Uhlenbeck process simulator."""

    def __init__(self, dim, mu=0., theta=0.15, sigma=0.2):
        """Initialization.

        :param int dim: dimension of the state.
        :param float mu: OU constant.
        :param float theta: OU constant.
        :param float sigma: OU constant.
        """
        self._mu = mu
        self._theta = theta
        self._sigma = sigma

        self._x = mu * np.ones(dim)

    def next(self):
        """Return the next state."""
        x = self._x
        w = np.random.randn(len(x))
        dx = self._theta * (self._mu - x) + self._sigma * w
        self._x += dx
        return self._x
