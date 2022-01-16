import gym
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


class TileCoder:

    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        # 每一个编码对应一个位置
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count > len(self.codebook):
            return hash(codeword) % self.features
        else:
            self.codebook[codeword] = count
            return count

    def __call__(self, floats=(), ints=()):
        """
        floats表示状态tuple(xt, vt),均归一化
        ints,表示tuple(a,)动作
        """
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        # features即存储当前状态动作对的特征
        features = []
        for layer in range(self.layers):
            # 对每个状态动作对编码
            # 编码为（layer, int(8xt + layer/8), int(8vt + 3*layer/8), A）
            codeword = (layer,) + tuple(
                int((f + (1 + dim * i) * layer) / self.layers) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features


class SarsaAgent:
    def __init__(self, env, layers=8, features=1893, gamma=1., learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low
        self.encoder = TileCoder(layers, features)
        self.w = np.zeros(features)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def encode(self, observation, action):
        # 对状态动作对作归一化处理
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        # 进行tile coding
        return self.encoder(states, actions)

    def get_q(self, observation, action):
        # 取出的features 是一组当前(S,A)在每一层中的位置，或者说是唯一编号
        # 其长度为tile coding的层数
        features = self.encode(observation, action)
        # 这里取出的features相当于是值编码，未取出的值均为0
        # 然后与权重向量作点乘，求和即为动作价值
        return self.w[features].sum()

    def decide(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            # 利用函数近似的方法直接求动作空间下的最优动作
            qs = [self.get_q(observation, action) for action in range(self.action_n)]
            return np.argmax(qs)

    def learn(self, observation, action, reward, next_observation, next_action, done):
        u = reward + (1. - done) * self.gamma * self.get_q(next_observation, next_action)
        td_error = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        # 更新权重向量
        self.w[features] += self.learning_rate * td_error


def play_sarsa(env, agent, train=False, render=False, episode_nums=200):
    episode_rewards = []
    for _ in range(episode_nums):
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
                agent.learn(observation, action, reward, next_observation, next_action, done)
            if done:
                break
            observation, action = next_observation, next_action
        episode_rewards.append(episode_reward)
    return episode_rewards


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    env = env.unwrapped

    print('观察空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('位置范围 = {}'.format((env.min_position, env.max_position)))
    print('速度范围 = {}'.format((-env.max_speed, env.max_speed)))
    print('目标位置 = {}'.format(env.goal_position))

    agent = SarsaAgent(env)

    episode_rewards = play_sarsa(env, agent, train=True, render=True)
    fig, ax = plt.subplots()
    ax.plot(episode_rewards, label="episode reward")
    ax.legend()
    fig.show()

    episode_rewards = play_sarsa(env, agent, train=False, episode_nums=100)
    print("averaging episode returns = {} / {} = {}".format(sum(episode_rewards), len(episode_rewards),
                                                            np.mean(episode_rewards)))
