import numpy as np

np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gym


class SARSAAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.2, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, next_action):
        u = reward + self.gamma * \
            self.q[next_state, next_action] * (1. - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)  # 终止状态时此步无意义
        if train:
            agent.learn(observation, action, reward, next_observation,
                        done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward


class ExpectedSARSAAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q = np.zeros((env.observation_space.n, env.action_space.n))
        self.action_n = env.action_space.n

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        v = (self.q[next_state].mean() * self.epsilon + \
             self.q[next_state].max() * (1. - self.epsilon))
        u = reward + self.gamma * v * (1. - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation,
                        done)
        if done:
            break
        observation = next_observation
    return episode_reward


class QLearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        u = reward + self.gamma * self.q[next_state].max() * (1. - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    env.seed(0)
    print('观察空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('状态数量 = {}'.format(env.observation_space.n))
    print('动作数量 = {}'.format(env.action_space.n))

    state = env.reset()
    taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
    print(taxirow, taxicol, passloc, destidx)
    print('的士位置 = {}'.format((taxirow, taxicol)))
    print('乘客位置 = {}'.format(env.unwrapped.locs[passloc]))
    print('目标位置 = {}'.format(env.unwrapped.locs[destidx]))
    env.render()

    env.step(0)

    env.render()

    agent = SARSAAgent(env)

    # 训练
    episodes = 3000
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_sarsa(env, agent, train=True)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)

    # 测试
    agent.epsilon = 0.  # 取消探索

    episode_rewards = [play_sarsa(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                         len(episode_rewards), np.mean(episode_rewards)))

    pd.DataFrame(agent.q)

    policy = np.eye(agent.action_n)[agent.q.argmax(axis=-1)]
    pd.DataFrame(policy)

    agent = ExpectedSARSAAgent(env)

    # 训练
    episodes = 5000
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)

    # 测试
    agent.epsilon = 0.  # 取消探索

    episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                         len(episode_rewards), np.mean(episode_rewards)))

    agent = QLearningAgent(env)

    # 训练
    episodes = 4000
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episode_rewards.append(episode_reward)

    plt.plot(episode_rewards)

    # 测试
    agent.epsilon = 0.  # 取消探索

    episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                         len(episode_rewards), np.mean(episode_rewards)))
