import gym
import numpy as np


class BespokeAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.06
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action

    def learn(self, *args):
        pass


def play_montecarlo(env, agent, render=True, train=False):
    """
    :param env: 表示环境
    :param agent: 表示智能体
    :param render: bool类型，是否图形化展示，如果为True，则要调用env.close()来关闭
    :param train: bool类型， 是否训练，如果为True，则调用agent.learn()，测试过程中应为False,保持agent不变
    :return:
    """
    # 记录回合总奖励，初始化为0
    episode_reward = 0.
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, agent, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    print(" 观测空间 = {}".format(env.observation_space))
    print(" 动作空间 = {}".format(env.action_space))
    print(" 观测范围 = {} ~ {}".format(env.observation_space.low, env.observation_space.high))
    print(" 动作数 = {}".format(env.action_space.n))

    # 设置随机化种子，可以让结果精确复现，一般情况下可以删去
    env.seed(0)
    agent = BespokeAgent(env)

    # episode_reward = play_montecarlo(env, agent, render=True)
    # print(' 回合奖励 = {}'.format(episode_reward))

    episode_reward = [play_montecarlo(env, agent) for _ in range(100)]
    print(' 平均回合奖励 = {}'.format(np.mean(episode_reward)))

    # 交互完毕，关闭图形化界面
    env.close()
