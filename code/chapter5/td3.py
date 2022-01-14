import gym
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

if __name__ == '__main__':
    env = gym.make("Taxi-v3")


    class DoubleQLearningAgent:
        def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
            self.gamma = gamma
            self.learning_rate = learning_rate
            self.epsilon = epsilon
            self.action_n = env.action_space.n
            self.q0 = np.zeros((env.observation_space.n, env.action_space.n))
            self.q1 = np.zeros((env.observation_space.n, env.action_space.n))

        def decide(self, state):
            if np.random.uniform() > self.epsilon:
                action = (self.q0 + self.q1)[state].argmax()
            else:
                action = np.random.randint(self.action_n)
            return action

        def learn(self, state, action, reward, next_state, done):
            if np.random.randint(2):
                self.q0, self.q1 = self.q1, self.q0
            a = self.q0[next_state].argmax()
            u = reward + self.gamma * self.q1[next_state, a] * (1. - done)
            td_error = u - self.q0[state, action]
            self.q0[state, action] += self.learning_rate * td_error


    agent = DoubleQLearningAgent(env)


    def play_double_QLeanring(env, agent, train=False, render=False, episode_nums=10000):
        episode_rewards = []
        for _ in range(episode_nums):
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


    episode_rewards = play_double_QLeanring(env, agent, train=True)
    plt.plot(episode_rewards)
    plt.show()
