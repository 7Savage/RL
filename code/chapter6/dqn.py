import numpy as np

np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
import gym
import tensorflow as tf

tf.random.set_seed(0)
from tensorflow import keras


class Chart:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)
        # plt.ion()

    def plot(self, episode_rewards):
        self.ax.clear()
        self.ax.plot(episode_rewards)
        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('episode reward')
        self.fig.show()


class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward',
                                            'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)


class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001,
                 replayer_capacity=10000, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)  # 经验回放

        self.evaluate_net = self.build_network(input_size=observation_dim,
                                               output_size=self.action_n, **net_kwargs)  # 评估网络
        self.target_net = self.build_network(input_size=observation_dim,
                                             output_size=self.action_n, **net_kwargs)  # 目标网络

        self.target_net.set_weights(self.evaluate_net.get_weights())  # 目标网络设置与评估网络相同的权重

    def build_network(self, input_size, hidden_sizes, output_size,
                      activation=tf.nn.relu, output_activation=None,
                      learning_rate=0.01):  # 构建网络
        model = keras.Sequential()  # 选择模型
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}  # 输入层，kwargs为可变参数字典
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, **kwargs))  # 添加隐藏层，Dense为全连接层，激活函数是relu
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation))  # 添加输出层，无激活函数
        optimizer = tf.optimizers.Adam(lr=learning_rate)  # 优化器为Adam
        model.compile(loss='mse', optimizer=optimizer)  # 编译，损失函数为均方差
        return model

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                            done)  # 存储经验 S A R S'

        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)  # 经验采样，Si Ai Ri Si'

        next_qs = self.target_net.predict(next_observations)  # 预测下一个动作价值函数
        next_max_qs = next_qs.max(axis=-1)  # 选择最大的动作价值函数
        us = rewards + self.gamma * (1. - dones) * next_max_qs  # 计算回报的估计值

        targets = self.evaluate_net.predict(observations)  # 只更新评估网络的权重
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=2)  # 训练，更新动作价值函数，verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果

        if done:  # 更新目标网络
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation):  # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        return np.argmax(qs)


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


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.seed(0)
    print('观测空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('位置范围 = {}'.format((env.unwrapped.min_position,
                              env.unwrapped.max_position)))
    print('速度范围 = {}'.format((-env.unwrapped.max_speed,
                              env.unwrapped.max_speed)))
    print('目标位置 = {}'.format(env.unwrapped.goal_position))

    net_kwargs = {'hidden_sizes': [64, 64], 'learning_rate': 0.001}
    agent = DQNAgent(env, net_kwargs=net_kwargs)

    # 训练
    episodes = 500
    episode_rewards = []
    chart = Chart()
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episode_rewards.append(episode_reward)
        chart.plot(episode_rewards)

    # 测试
    agent.epsilon = 0.  # 取消探索
    episode_rewards = [play_qlearning(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                         len(episode_rewards), np.mean(episode_rewards)))
