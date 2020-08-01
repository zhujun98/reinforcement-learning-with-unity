import platform

import numpy as np

import unityagents as ua

from .utilities import check_environment


if __name__ == "__main__":

    if platform.system() == "Linux":
        executable_file = "Banana_Linux/Banana.x86_64"
    else:
        raise RuntimeError

    env = ua.UnityEnvironment(file_name=executable_file)

    brain_name = check_environment(env)
