import argparse

import torch

from papi_rl import PapiAgent, PapiEnv


def main(args):
    agent = PapiAgent()
    agent.load_state_dict(torch.load("papi_agent.pt"))

    env = PapiEnv(normal_speed=False)
    obs = env.reset()
    env.render()

    done = False
    while not done:
        with torch.no_grad():
            action_probs, _ = agent(torch.from_numpy(obs).float())
        obs, reward, done, _ = env.step(torch.argmax(action_probs).cpu().item())
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
