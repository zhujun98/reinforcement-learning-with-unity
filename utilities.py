"""
Copyright (c) 2020 Jun Zhu
"""
import copy

import numpy as np


def check_environment(env, *, return_n_agents=False):
    """Check environment.

    :param UnityEnvironment env: unity environment.
    :param bool return_n_agents: True for returning also the number of agents.
        For backward compatibility.
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    print("Brain name: ", brain_name)
    print("Brain: ", brain)

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents in the environment
    n_agents = len(env_info.agents)
    print('Number of agents:', n_agents)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    if return_n_agents:
        return brain_name, state_size, action_size, n_agents
    return brain_name, state_size, action_size


def play(env, brain_name=None, agent=None, continuous=False, repeats=1):
    n_agents = len(env.reset(train_mode=False)[brain_name].agents)
    if agent is None:
        for i in range(repeats):
            scores = [0] * n_agents

            # play randomly
            brain = env.brains[brain_name]
            action_size = brain.vector_action_space_size
            env.reset(train_mode=False)
            while True:
                if continuous:
                    actions = np.clip(
                        np.random.randn(n_agents, action_size), -1., 1.)
                else:
                    actions = np.random.randint(action_size)

                env_info = env.step(actions)[brain_name]
                for i_a in range(n_agents):
                    scores[i_a] += env_info.rewards[i_a]

                if env_info.local_done[0]:
                    break
            print(f"Score of play {i+1:02d}: {max(scores):>12.4f}")
    else:
        for i in range(repeats):
            # use a trained agent
            score = agent.play(env)
            print(f"Score of play {i+1:02d}: {score:>12.4f}")


def plot_score_history(ax, scores, target_score):
    x = np.arange(len(scores)) + 1
    ax.plot(x, scores, alpha=180, label='score history')
    ax.plot(x[99:], np.convolve(scores, np.ones(100) / 100, 'valid'),
            label='score history (moving averaged)')
    ax.plot(x, target_score * np.ones_like(x), '--', label='reward (target)')
    ax.set_xlabel("Episode", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.legend()


def plot_losses(ax, loss1, label1, loss2=None, label2=None, *, downsampling=1.):
    stride = int(1./downsampling)

    ax.plot(loss1[::stride], color='tab:blue')
    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel(label1, color='tab:blue', fontsize=16)
    ax.tick_params(axis='y', labelcolor='tab:blue')

    if loss2 is not None:
        ax2 = ax.twinx()
        ax2.plot(loss2[::stride], color='tab:orange', alpha=0.5)
        ax2.set_ylabel(label2, color='tab:orange', fontsize=16)
        ax2.tick_params(axis='y', labelcolor='tab:orange')


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


def copy_nn(src, dst):
    """Copy the parameters of a source network to the target."""
    dst.load_state_dict(copy.deepcopy(src.state_dict()))


def soft_update_nn(src, dst, tau):
    """Apply soft update to a target network.

    :param torch.nn.Module src: src Neural network model.
    :param torch.nn.Module dst: dst Neural network model.
    :param float tau: soft update factor.
    """
    for src_param, dst_param in zip(src.parameters(), dst.parameters()):
        dst_param.data.copy_(
            tau * src_param.data + (1. - tau) * dst_param.data)
