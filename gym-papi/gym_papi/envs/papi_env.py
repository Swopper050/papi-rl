import gym


class PapiEnv(gym.Env):
    """
    TODO: Add description of environment
    """

    metadata = {
        "render.modes": ["human"],
        "video.frames_per_second": 50,
    }

    def __init__(self):
        self.observation_space = None
        self.action_space = None
        self.reset()

    def step(self, action):
        """
        Perform action in the environment
        """
        pass

    def reset(self):
        """
        Should reset the environment
        """
        pass

    def render(self, mode="human"):
        """
        Should render the environment
        """
        pass

    def close(self):
        """
        Should close the environment
        """
        pass
