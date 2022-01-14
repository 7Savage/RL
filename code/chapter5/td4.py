import numpy as np
import gym
import matplotlib.pyplot as plt

np.random.seed(0)


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


if __name__ == '__main__':
    env = gym.make("Taxi-v3")


    class SarsaLambdaAgent(SarsaAgent):
        def __init__(self, env, lmd=0.5, beta=1.0, gamma=0.9, learning_rate=0.1, epsilon=0.01):
            super().__init__(env, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
            self.lmd = lmd
            self.beta = beta
            self.e = np.zeros((env.observation_space.n, env.action_space.n))

        def learn(self, state, action, reward, next_state, next_action, done):
            # 更新资格迹
            self.e *= (self.lmd * self.gamma)
            self.e[state, action] = 1. + self.beta * self.e[state, action]
            # 更新价值
            u = reward + self.gamma * self.q[next_state, next_action] * (1 - done)
            td_error = u - self.q[state, action]
            self.q += self.learning_rate * self.e * td_error

            # 为下个回合初始化资格迹
            if done:
                self.e *= 0.


    agent = SarsaLambdaAgent(env)


    def play_sarsa_lambda(env, agent, train=False, render=False, episode_nums=10000):
        episode_rewards = []
        for _ in range(episode_nums):
            observation = env.reset()
            episode_reward = 0
            action = agent.decide(observation)
            while True:
                if render:
                    env.render()
                next_observation, reward, done, _ = env.step(action)
                episode_reward += reward
                next_action = agent.decide(next_observation)
                if train:
                    agent.learn(observation, action, reward, next_observation, next_action, done)
                if done:
                    break
                observation, action = next_observation, next_action

            episode_rewards.append(episode_reward)
        return episode_rewards


    episode_rewards = play_sarsa_lambda(env, agent, train=True)
    plt.plot(episode_rewards)
    plt.show()

    episode_rewards = play_sarsa_lambda(env, agent, episode_nums=100)
    print("averaging episode returns = {} / {} = {}".format(sum(episode_rewards), len(episode_rewards),
                                                            np.mean(episode_rewards)))
