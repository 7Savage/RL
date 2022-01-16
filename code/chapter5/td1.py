import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import gym

np.random.seed(0)

if __name__ == '__main__':
    env = gym.make("Taxi-v3")
    env.seed(0)
    print('观察空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('状态数量 = {}'.format(env.observation_space.n))
    print('动作数量 = {}'.format(env.action_space.n))

    state = env.reset()
    taxi_row, taxi_col, pass_loc, dest_id = env.unwrapped.decode(state)
    print(taxi_row, taxi_col, pass_loc, dest_id)
    print("出租车位置 = {}".format((taxi_row, taxi_col)))
    print("乘客的位置 = {}".format(pass_loc))
    print("目的地位置 = {}".format(dest_id))

    env.render()
    print("initial state : {}".format(state))


    class SarsaAgent:
        """
        Sarsa算法
        epsilon贪心策略选择动作，动作不显式地用policy来存储了，而是直接从状态价值中选取最大的
        策略评估过程，即将公式套进去即可
        """

        def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
            self.gamma = gamma
            self.learning_rate = learning_rate
            self.epsilon = epsilon
            self.action_n = env.action_space.n
            self.q = np.zeros((env.observation_space.n, env.action_space.n))

        def decide(self, state):
            if np.random.uniform() > self.epsilon:
                # 利用现有最大的价值选择动作
                action = self.q[state].argmax()
            else:
                # 有一定的概率探索其它动作
                action = np.random.randint(self.action_n)
            return action

        def learn(self, state, action, reward, next_state, done, next_action):
            # Sarsa 直接利用定义，用下一个时刻的状态动作对(St+1, At+1)可以更新当前的回报Ut
            # 那么就需要知道当前的状态动作对，和奖励
            # Sarsa 也因此而得名
            u = reward + self.gamma * self.q[next_state, next_action] * (1. - done)
            td_error = u - self.q[state, action]
            self.q[state, action] += self.learning_rate * td_error


    agent = SarsaAgent(env)


    def play_sarsa(env, agent, train=False, render=False, episode_nums=10000):
        episode_rewards = []
        for _ in range(episode_nums):
            # 每个新的回合都需要重置
            observation = env.reset()
            action = agent.decide(observation)
            episode_reward = 0
            while True:
                if render:
                    env.render()
                next_observation, reward, done, _ = env.step(action)
                episode_reward += reward
                next_action = agent.decide(next_observation)
                if train:
                    agent.learn(observation, action, reward, next_observation, done, next_action)
                if done:
                    break
                observation, action = next_observation, next_action

            episode_rewards.append(episode_reward)

        return episode_rewards


    episode_rewards = play_sarsa(env, agent, train=True, episode_nums=5000)
    plt.plot(episode_rewards)
    plt.title("Sarsa")
    plt.show()
    # np.savetxt("q.csv", agent.q, fmt="%.5e", delimiter=",")

    # 训练完毕，测试策略

    agent.epsilon = 0
    episode_rewards = play_sarsa(env, agent, episode_nums=100)
    print("averaging episode returns = {} / {} = {}".format(sum(episode_rewards), len(episode_rewards),
                                                            np.mean(episode_rewards)))

    pd.DataFrame(agent.q)
    # 这个操作相当于把数字标签改成独热编码
    policy = np.eye(agent.action_n)[agent.q.argmax(axis=-1)]
    pd.DataFrame(policy)
