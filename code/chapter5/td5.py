import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import gym

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


class MultiStepSarsaAgent(SarsaAgent):
    def __init__(self, env, n_step=2, gamma=0.9, learning_rate=0.1, epsilon=.01):
        super().__init__(env, gamma, learning_rate, epsilon)
        self.env = env
        self.n_step = n_step
        # 定义n步轨迹，每一步都包括S A R' S'
        self.trajectory_shape = (self.n_step, 4)
        self.trajectory = np.zeros(self.trajectory_shape, dtype=int)
        self.reward_T = 0

    def __calc_episode_reward(self, n_reward):
        coe = np.power(self.gamma, np.arange(len(n_reward)))
        G = np.dot(coe, n_reward)
        return G

    def exploring_start(self):
        # 每个回合开始，保轨迹为空
        self.trajectory *= 0
        state = self.env.reset()
        state_done = False
        for i in range(self.n_step):
            # 用动作价值估计q确定的epsilon贪心策略生成n步轨迹
            action = self.decide(state)
            if state_done:
                # 若遇到终止状态，则令后续的奖励均为0， 状态均为终止状态, 动作仍然可以随机选择，
                # 但是奖励为0，这里奖励为0不知道是否合适，或许更小才合适
                self.trajectory[i] = int(state), int(action), int(self.reward_T), int(state_done)
                continue
            next_state, reward, done, _ = self.env.step(action)
            self.trajectory[i] = int(state), int(action), int(reward), int(state_done)
            state, state_done = next_state, done
        return state, state_done

    def update_trajectory(self, new_row):
        self.trajectory = np.roll(self.trajectory, -self.trajectory_shape[1])
        self.trajectory[-1] = new_row

    def store_q(self, fname):
        self.q = np.loadtxt(fname, delimiter=",")

    def learn(self, state, action, n_reward, next_state, next_action, done):
        """
        n步Sarsa算法
        首先需要根据策略生成n步轨迹，前n步的轨迹作为窗口，此后在此基础上进行单步移动，即步长为1
        此处的n_reward 指的是前窗口内的奖励，
        next_state 和next_action指的是t+n步时的状态动作对，如果已经是结束状态，
        则使用（1-done）使q(St+n, At+n)为0
        也就是说后面的状态价值都为0 差分更新目标U=G
        """
        Gtn = self.__calc_episode_reward(n_reward)
        u = Gtn + (1. - done) * (np.power(self.gamma, len(n_reward)) * self.q[next_state, next_action])
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error


def play_multi_step_sarsa(env, agent, train=False, render=False, episode_nums=10000):
    episode_rewards = []
    for _ in range(episode_nums):
        # 生成n步轨迹，
        episode_reward = 0
        if train:
            # 先探索n步
            next_state, state_done = agent.exploring_start()
            episode_reward += np.sum(agent.trajectory[:, 2])
            while True:
                # 若要更新的时刻的结束标记为True，则退出
                if agent.trajectory[0, -1]:
                    break
                # 根据q(St+n, )选择At+n
                next_action = agent.decide(next_state)
                # 更新价值
                agent.learn(state=agent.trajectory[0, 0],
                            action=agent.trajectory[0, 1],
                            n_reward=agent.trajectory[:, 2],
                            next_state=next_state,
                            next_action=next_action,
                            done=state_done)
                # 判断St+n 是不是终止状态,并获取下个状态
                if state_done:
                    reward = agent.reward_T
                    new_state, done = next_state, state_done
                else:
                    # 如果最后这个状态不是终止状态，则执行下一步
                    new_state, reward, done, _ = env.step(next_action)
                    # 根据需要渲染
                    if render:
                        env.render()
                agent.update_trajectory([next_state, next_action, reward, state_done])
                next_state, state_done = new_state, done
                episode_reward += reward

        episode_rewards.append(episode_reward)
    return episode_rewards


def play_test(env, agent, episode_nums=100):
    episode_rewards = []
    # 取消随机性
    agent.epsilon = 0
    for _ in range(episode_nums):
        state = env.reset()
        action = agent.decide(state)
        episode_reward = 0
        while True:
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            next_action = agent.decide(next_state)
	    if done:
                break
            action = next_action
        episode_rewards.append(episode_reward)
    return episode_rewards


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

    ms_agent = MultiStepSarsaAgent(env, n_step=3)

    episode_rewards = play_multi_step_sarsa(env, ms_agent, train=True, episode_nums=5000)

    plt.plot(episode_rewards)
    plt.title("{}-step Sarsa".format(ms_agent.n_step))
    plt.show()
    # np.savetxt("ms_q.csv", ms_agent.q, fmt="%.5e", delimiter=",")

    # ms_agent.store_q("ms_q.csv")

    episode_rewards = play_test(env, ms_agent)
    print("averaging episode returns = {} / {} = {}".format(sum(episode_rewards), len(episode_rewards),
                                                            np.mean(episode_rewards)))
