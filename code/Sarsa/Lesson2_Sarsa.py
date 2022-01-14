import gym
import numpy as np

import time


# agent
class SarsaAgent:
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        # 动作维度， 有几个动作可选
        self.act_n = act_n
        # 学习率
        self.lr = learning_rate
        # reward的折扣因子
        self.gamma = gamma
        # 贪心机率
        self.epsilon = e_greed
        # 初始化Q值
        self.Q = np.zeros((obs_n, act_n))

    def sample(self, obs):
        """
        根据输入的观察值，采样输出的动作值，带探索能力（贪心策略）
        :param obs:
        :return:
        """
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    def predict(self, obs):
        """
        根据输入的观察值，预测输出的动作值
        :param obs:
        :return:
        """
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        action = np.random.choice(action_list)
        return action

    def learn(self, obs, action, reward, next_obs, next_action, done):
        """
        学习方法，也就是更新Q-table的方法
        :param obs: s_t
        :param action: a_t
        :param reward: r_t+1
        :param next_obs: s_t+1
        :param next_action: a_t+1
        :param done: episode 是否结束
        :return:
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            # # 与Q leaning 唯一不同的地方
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)

    def save(self):
        """
        保存Q表格到文件
        :return:
        """
        npy_file = "./q_table.npy"
        np.save(npy_file, self.Q)
        print(npy_file + " saved.")

    def restore(self, npy_file="./q_table.npy"):
        """
        加载Q表格
        :return:
        """
        self.Q = np.load(npy_file)
        print(npy_file + " loaded.")


def run_episode(env, agent, render=False):
    # 记录每个episode走了多少步
    total_steps = 0
    total_reward = 0

    # 重置环境
    obs = env.reset()
    action = agent.sample(obs)

    while True:
        # 与环境交互
        next_obs, reward, done, _ = env.step(action)
        # 根据算法选择一个动作
        next_action = agent.sample(next_obs)

        # 训练Sarsa算法
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs

        total_reward += reward
        total_steps += 1
        if render:
            # 渲染一帧图形
            env.render()
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs

        if done:
            break
    return total_reward


if __name__ == '__main__':
    # 选择环境
    env = gym.make("CliffWalking-v0")
    # 创建agent 实例， 输入超参数
    agent = SarsaAgent(obs_n=env.observation_space.n,
                       act_n=env.action_space.n,
                       learning_rate=0.1,
                       gamma=0.9,
                       e_greed=0.1)

    npy_file = "./q_table.npy"
    try:
        agent.restore(npy_file)
    except Exception as e:
        print(Exception)

    # 训练500个episode, 打印每个episode的分数
    for episode in range(50):
        ep_reward, ep_steps = run_episode(env, agent, True)
        print("Episode %s: steps = %s , reward = %.1f" % (episode, ep_steps, ep_reward))

    agent.save()

    # 全部训练结束，查看算法效果
    test_reward = test_episode(env, agent)
    print("test reward = %.1f" % test_reward)
