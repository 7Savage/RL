import gym
import numpy as np
import matplotlib.pyplot as plt
import fa2

np.random.seed(0)


class SarsaLambdaAgent(fa2.SarsaAgent):
    def __init__(self, env, layers=8, features=1893, gamma=1., learning_rate=0.03, epsilon=0.001, lambd=0.9):
        super().__init__(env, layers, features, gamma, learning_rate, epsilon)
        self.lambd = lambd
        self.z = np.zeros(features)

    def learn(self, observation, action, reward, next_observation, next_action, done):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(next_observation, next_action))
            self.z *= (self.gamma * self.lambd)
            features = self.encode(observation, action)
            # 替换迹
            self.z[features] = 1.
        td_error = u - self.get_q(observation, action)
        self.w += self.learning_rate * td_error * self.z
        if done:
            # 为下个回合初始化资格迹
            self.z = np.zeros_like(self.z)


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    env = env.unwrapped

    agent = SarsaLambdaAgent(env)
    episode_rewards = fa2.play_sarsa(env, agent, train=True, render=False, episode_nums=200)
    fig, ax = plt.subplots()
    ax.plot(episode_rewards, label="episode reward")
    ax.legend()
    fig.show()

    episode_rewards = fa2.play_sarsa(env, agent, train=False, episode_nums=100)
    print("averaging episode returns = {} / {} = {}".format(sum(episode_rewards), len(episode_rewards),
                                                            np.mean(episode_rewards)))
