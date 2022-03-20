import multiprocessing as mp
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from papi_rl.papi_env import N_FRAMES_PER_OBS, PapiEnv


def train_agent(n_workers=(os.cpu_count() - 1), n_epochs=1000):
    master_papi = PapiAgent()
    master_papi.share_memory()
    processes = []

    counter = mp.Value("i", 0)
    worker(None, master_papi, counter, n_epochs)
    for i in range(n_workers):
        p = mp.Process(target=worker, args=(i, master_papi, counter, n_epochs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for p in processes:
        p.terminate()


def worker(t, agent, counter, n_epochs):
    worker_env = PapiEnv(normal_speed=False)
    worker_env.reset()

    worker_opt = optim.Adam(lr=1e-4, params=agent.parameters())

    for i in range(n_epochs):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, agent)
        update_params(worker_opt, values, logprobs, rewards)
        counter.value = counter.value + 1


def run_episode(env, agent):
    values, logprobs, rewards = [], [], []
    done = False

    obs = env.get_observation()
    G = torch.Tensor([0])
    while not done:
        obs = torch.from_numpy(env.get_observation()).float()
        action_probs, value_prediction = agent(obs)
        action = torch.distributions.Categorical(logits=action_probs.view(-1)).sample()

        obs, reward, done, _ = env.step(action.detach().numpy().item())
        env.render()

        values.append(value_prediction)
        logprobs.append(action_probs.view(-1)[action])
        rewards.append(reward)

    __import__("pdb").set_trace()
    return (
        torch.stack(values).flip(dims=(0,)).view(-1),
        torch.stack(logprobs).flip(dims=(0,)).view(-1),
        torch.Tensor(rewards).flip(dims=(0,)).view(-1),
    )


def update_params(optimizer, values, logprobs, rewards, clc=0.1, gamma=0.95):
    __import__("pdb").set_trace()


class PapiAgent(nn.Module):
    """
    Actor-Critic architecture that will learn to play the game.
    """

    def __init__(self):
        super(PapiAgent, self).__init__()
        self.conv1 = nn.Conv2d(N_FRAMES_PER_OBS, 10, 100, stride=10)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 10)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5980, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor_fc = nn.Linear(128, 4)
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, x):
        has_batch = len(x.shape) == 4
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if has_batch:
            x = self.flatten(x)
        else:
            x = x.flatten()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = F.log_softmax(self.actor_fc(x), dim=0)
        values = self.critic_fc(x.detach())
        return actions, values

    def act(self):
        pass
