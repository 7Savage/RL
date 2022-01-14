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


    class ExpectedSarsaAgent:
        def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
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

        def learn(self, state, action, reward, next_state, done):
            """
            Expected Sarsa算法不同于Sarsa算法的地方在于估计回报的方法
            它不使用基于动作价值的时序差分目标，而使用基于状态价值的时序差分目标
            即它不利用q(St+1, At+1)而是利用v(St+1), 由bellman期望方程，
            状态价值等于对动作价值的期望
            由于我们的策略是epsilon贪心策略，所以此处的状态价值应该按照下面的方法来求解
            """
            # 所有动作都有epsilon的概率被选到，所以用self.q[next_state].mean() * epsilon
            # 但是价值最大的动作可以被选到的概率为(1-epsilon)
            # 所以总得价值应该为下述两者之和
            v = (self.q[next_state].mean() * self.epsilon +
                 self.q[next_state].max() * (1. - self.epsilon))

            # 基于状态价值算回报
            u = reward + self.gamma * v * (1. - done)
            # 计算误差
            td_error = u - self.q[state, action]
            # 更新动作价值
            self.q[state, action] += self.learning_rate * td_error


    class QLearningAgent:

        def __init__(self, env, gamma=0.9, learning_rate=.1, epsilon=0.01):
            self.gamma = gamma
            self.learning_rate = learning_rate
            self.epsilon = epsilon
            self.action_n = env.action_space.n
            self.q = np.zeros((env.state_space.n, env.action_space.n))

        def decide(self, state):
            if np.random.uniform() > self.epsilon:
                action = self.q[state].argmax()
            else:
                action = np.random.randint(self.action_n)
            return action

        def learn(self, state, action, reward, next_state, done):
            """"
            q learning 认为，当前时刻的回报Ut，可以直接根据更新后的策略来取出St+1下的最优动作
            这样的更新可以更加接近与最优价值. 因此Q学习的更新式不是利用当前的策略（上面的epsilon策略）
            而是利用一个已知的但是不一定要使用的确定性策略来更新动作价值。
            所以Q learning是一个异策算法
            """
            u = reward + self.gamma * self.q[next_state].max() * (1. - done)
            td_error = u - self.q[state, action]
            self.q[state, action] += (self.learning_rate * td_error)


    agent = ExpectedSarsaAgent(env)


    def play_qlearning(env, agent, train=False, render=False, episode_nums=10000):
        episode_rewards = []
        for _ in range(episode_nums):
            # 每个新的回合都需要重置
            observation = env.reset()
            episode_reward = 0
            while True:
                if render:
                    env.render()
                action = agent.decide(observation)
                next_observation, reward, done, _ = env.step(action)
                episode_reward += reward
                if train:
                    agent.learn(observation, action, reward, next_observation, done)
                if done:
                    break
                observation = next_observation

            episode_rewards.append(episode_reward)

        return episode_rewards


    episode_rewards = play_qlearning(env, agent, train=True)

    plt.plot(episode_rewards)
    plt.show()

    # 训练完毕，测试策略
    agent.epsilon = 0
    episode_rewards = play_qlearning(env, agent, episode_nums=100)
    print("averaging episode returns = {} / {} = {}".format(sum(episode_rewards), len(episode_rewards),
                                                            np.mean(episode_rewards)))
