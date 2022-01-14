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


    class MultiStepSarsaAgent(SarsaAgent):
        def __init__(self, env, n_step=2, gamma=0.9, learning_rate=0.1, epsilon=.01):
            super().__init__(env, gamma, learning_rate, epsilon)
            self.env = env
            self.n_step = n_step
            # 定义n步轨迹，每一步都包括S A R' S'
            self.trajectory_shape = (self.n_step, 5)
            self.trajectory = np.zeros(self.trajectory_shape, dtype=int)

        def __calc_episode_reward(self, n_reward):
            coe = np.power(self.gamma, np.arange(len(n_reward)))
            G = np.dot(coe, n_reward)
            return G

        def exploring_start(self):
            # 每个回合开始，保轨迹为空
            self.trajectory *= 0
            reward_T = 0
            state = self.env.reset()
            done = False
            for i in range(self.n_step):
                # 用动作价值估计q确定的epsilon贪心策略生成n步轨迹
                action = self.decide(state)
                if done:
                    # 若遇到终止状态，则令后续的奖励均为0， 状态均为终止状态, 动作仍然可以随机选择，
                    # 但是奖励为0，这里奖励为0不知道是否合适，或许更小才合适
                    self.trajectory[i] = int(state), int(action), int(reward_T), int(state), int(done)
                    continue
                next_state, reward, done, _ = self.env.step(action)
                self.trajectory[i] = int(state), int(action), int(reward), int(next_state), int(done)
                state = next_state

        def update_trajectory(self, state, action, reward, new_observation, done):
            self.trajectory = np.roll(self.trajectory, -self.trajectory_shape[1])
            self.trajectory[-1] = state, action, reward, new_observation, done

        def learn(self, state, action, next_state, next_action, done):
            """
            n步Sarsa算法
            首先需要根据策略生成n步轨迹，前n步的轨迹作为窗口，此后在此基础上进行单步移动，即步长为1
            此处的n_reward 指的是前窗口内的奖励，
            next_state 和next_action指的是t+n步时的状态动作对，如果已经是结束状态，则（1-done）使之为0
            也就是说后面的状态价值都为0

            """
            n_reward = self.trajectory[:, 2]
            Gtn = self.__calc_episode_reward(n_reward)
            u = Gtn + (np.power(self.gamma, len(n_reward)) * self.q[next_state, next_action] * (1. - done))
            td_error = u - self.q[state, action]
            self.q[state, action] += self.learning_rate * td_error


    ms_agent = MultiStepSarsaAgent(env)


    def play_multi_step_sarsa(env, agent, n=2, train=False, render=False, episode_nums=10000):
        episode_rewards = []
        for _ in range(episode_nums):
            # 生成n步轨迹，
            history = np.zeros((n, 5))
            observation = env.reset()
            episode_reward = 0
            done = False
            if train:
                # 先探索n步
                for i in range(n):
                    # 用动作价值估计q确定的epsilon贪心策略生成n步轨迹
                    action = agent.decide(int(observation))
                    if done:
                        # 若遇到终止状态，则令后续的奖励均为0， 状态均为终止状态, 动作仍然可以随机选择，
                        # 但是奖励为0，这里奖励为0不知道是否合适，或许更小才合适
                        history[i] = observation, action, -10, observation, int(done)
                        continue
                    next_observation, reward, done, _ = env.step(action)
                    episode_reward += reward
                    history[i] = observation, action, reward, next_observation, int(done)
                    observation = next_observation

            while True:
                if history[0, -1]:
                    break
                # 拿到奖励池
                reward_pool = history[:, 2]
                next_observation = history[-1, 3]
                # 根据q(St+n, )选择At+n
                next_action = agent.decide(int(next_observation))
                # 更新价值
                if train:
                    agent.learn(state=int(history[0, 0]), action=int(history[0, 1]), n_reward=reward_pool,
                                next_state=int(next_observation), next_action=int(next_action), done=history[-1, -1])
                # 判断St+n 是不是终止状态
                is_done = history[-1, 4]
                if is_done:
                    reward = -10
                    new_observation = next_observation
                else:
                    # 如果最后这个状态不是终止状态，则执行下一步
                    new_observation, reward, done, _ = env.step(next_action)
                    # 根据需要渲染
                    if render:
                        env.render()
                episode_reward += reward

            episode_rewards.append(episode_reward)
        return episode_rewards


    episode_rewards = play_multi_step_sarsa(env, ms_agent, n=5, train=True, episode_nums=5000)

    plt.plot(episode_rewards)
    plt.show()

    episode_rewards = play_multi_step_sarsa(env, ms_agent, episode_nums=100)
    print("averaging episode returns = {} / {} = {}".format(sum(episode_rewards), len(episode_rewards),
                                                            np.mean(episode_rewards)))
