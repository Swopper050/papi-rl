import argparse
import random

from papi_rl import PapiEnv


def main(args):
    env = PapiEnv(normal_speed=False)

    obs = env.reset()
    env.render()

    done = False
    while not done:
        action = random.randint(0, 3)
        obs, reward, done, _ = env.step(action)
        __import__("pdb").set_trace()
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
