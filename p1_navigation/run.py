import platform

import numpy as np

import unityagents as ua

from utilities import check_environment


if __name__ == "__main__":

    if platform.system() == "Linux":
        executable_file = "Banana_Linux/Banana.x86_64"
    else:
        raise RuntimeError

    env = ua.UnityEnvironment(file_name=executable_file)

    brain_name, action_size = check_environment(env)

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = np.random.randint(action_size)  # select an action
        env_info = env.step(action)[
            brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))
