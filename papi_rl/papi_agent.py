import multiprocessing as mp
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from papi_rl.papi_env import N_FRAMES_PER_OBS, SCREEN_HEIGHT, SCREEN_WIDTH, PapiEnv


def train_agent(n_workers=(os.cpu_count() - 1), n_epochs=1000):
    papi_agent = PapiAgent()
    env = PapiEnv(normal_speed=False)
    env.reset()

    optimizer = optim.Adam(lr=1e-4, params=papi_agent.parameters())
    for i in range(n_epochs):
        print(f"At epoch {i}")
        optimizer.zero_grad()
        values, logprobs, rewards, G = run_episode(env, papi_agent)
        update_params(optimizer, values, logprobs, rewards, G)

        if i % 100 == 0:
            torch.save(papi_agent.state_dict(), "papi_agent.pt")


def run_episode(env, agent, n_steps=10):
    values, logprobs, rewards = [], [], []
    done = False

    obs = env.get_observation()
    G = torch.Tensor([0])
    n = 0
    while not done and n < n_steps:
        obs = torch.from_numpy(env.get_observation()).float()
        action_probs, value_prediction = agent(obs)
        action = torch.distributions.Categorical(logits=action_probs.view(-1)).sample()

        obs, reward, done, _ = env.step(action.detach().cpu().numpy().item())
        env.render()

        values.append(value_prediction)
        logprobs.append(action_probs.view(-1)[action])
        rewards.append(reward)
        if not done:
            G = value_prediction.detach()

        n += 1
        if done:
            env.reset()

    return (
        torch.stack(values).flip(dims=(0,)).view(-1),
        torch.stack(logprobs).flip(dims=(0,)).view(-1),
        torch.Tensor(rewards).flip(dims=(0,)).view(-1),
        G,
    )


def update_params(
    optimizer, values, logprobs, rewards, G, critic_loss_weight=0.1, gamma=0.95
):
    returns = []
    prev_return = G
    for r in range(rewards.shape[0]):
        returns.append(rewards[r] + gamma * prev_return)

    returns = torch.stack(returns).view(-1)
    returns = F.normalize(returns, dim=0)

    actor_loss = -1 * logprobs * (returns - values.detach())
    critic_loss = torch.pow(values - returns, 2)
    loss = actor_loss.sum() + critic_loss_weight * critic_loss.sum()
    loss.backward()
    optimizer.step()


class PapiAgent(nn.Module):
    """
    Actor-Critic architecture that will learn to play the game.
    """

    def __init__(self):
        super(PapiAgent, self).__init__()
        self.conv1 = nn.Conv2d(N_FRAMES_PER_OBS, 5, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 10)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(565490, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor_fc = nn.Linear(128, 4)
        self.critic_fc = nn.Linear(128, 1)
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.to(self.device)

    def forward(self, x):
        has_batch = len(x.shape) == 4
        x = x.to(self.device)
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
