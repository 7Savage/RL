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

    def plot(self, episode_rewards):
        self.ax.clear()
        self.ax.plot(episode_rewards)
        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('episode reward')
        self.fig.show()


class VPGAgent:
    def __init__(self, env, policy_kwargs, baseline_kwargs=None,
                 gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma

        self.trajectory = []

        self.policy_net = self.build_network(output_size=self.action_n,
                                             output_activation=tf.nn.softmax,
                                             loss=tf.losses.categorical_crossentropy,  # 损失函数是交叉熵
                                             **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_network(**baseline_kwargs)

    def build_network(self, hidden_sizes, output_size=1,
                      activation=tf.nn.relu, output_activation=None,
                      use_bias=False, loss=tf.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for hidden_size in hidden_sizes:
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, use_bias=use_bias))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation, use_bias=use_bias))
        optimizer = tf.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    #  根据神经网络，输入 S ，输出 A
    def decide(self, observation):
        probs = self.policy_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, done):
        self.trajectory.append((observation, action, reward))

        if done:
            df = pd.DataFrame(self.trajectory,
                              columns=['observation', 'action', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()  # 对折扣奖励累和
            df['psi'] = df['discounted_return']

            x = np.stack(df['observation'])  # x表示观测的四个分量：位置、速度、角度、角速度
            #  如果该对象有 baseline_net，则返回 true
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x)
                df['psi'] -= (df['baseline'] * df['discount'])
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)

            sample_weight = df['psi'].values[:, np.newaxis]  # 采样权重
            y = np.eye(self.action_n)[df['action']]  # y表示采取的动作
            self.policy_net.fit(x, y, sample_weight=sample_weight, verbose=0)

            self.trajectory = []  # 为下一回合初始化经验列表


def play_montecarlo(env, agent, render=False, train=False):
    observation = env.reset()  # 状态 S
    episode_reward = 0.
    while True:
        if render:
            env.render()
        action = agent.decide(observation)  # 动作 A
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)

    # 不带基线的简单策略梯度算法
    policy_kwargs = {'hidden_sizes': [], 'learning_rate': 0.005}
    agent = VPGAgent(env, policy_kwargs=policy_kwargs)

    # 训练
    episodes = 1000
    episode_rewards = []
    chart = Chart()
    for episode in range(episodes):
        episode_reward = play_montecarlo(env, agent, train=True)
        episode_rewards.append(episode_reward)
        chart.plot(episode_rewards)

    # 测试
    episode_rewards = [play_montecarlo(env, agent, train=False)
                       for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                         len(episode_rewards), np.mean(episode_rewards)))
