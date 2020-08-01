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

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    return brain_name, action_size


def play(env, brain_name=None, agent=None):
    score = 0
    if agent is None:
        # play randomly
        brain = env.brains[brain_name]
        action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=False)[brain_name]
        while True:
            action = np.random.randint(action_size)
            env_info = env.step(action)[brain_name]
            state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            if done:
                break
    else:
        # use a trained agent
        agent.play(env)

    print("Score: {}".format(score))
