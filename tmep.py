import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

# 定义神经网络模型
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

# 定义策略梯度算法
class PolicyGradient:
    def __init__(self, input_dim, output_dim, lr=0.001):
        self.policy_net = PolicyNet(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        action = probs.multinomial(num_samples=1)
        return action.item()

    def update(self, rewards, log_probs):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            discount = 1
            for k in range(t, len(rewards)):
                Gt += discount * rewards[k]
                discount *= 0.99
            discounted_rewards.append(Gt)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        log_probs = torch.stack(log_probs)
        loss = -torch.mean(log_probs * discounted_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 定义训练函数
def train(env, agent, episodes):
    for episode in range(episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            log_prob = torch.log(agent.policy_net(torch.FloatTensor(state)))
            log_probs.append(log_prob)
            rewards.append(reward)
        agent.update(rewards, log_probs)
        print('Episode %d: Reward = %d' % (episode, np.sum(rewards)))

# 加载CIFAR-10数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化强化学习环境和智能体
env = train_loader
agent = PolicyGradient(3072, 10)

# 训练智能体
train(env, agent, episodes=100)
