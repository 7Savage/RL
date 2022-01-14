import numpy as np
import matplotlib.pyplot as plt
import gym

np.random.seed(0)


def ob2state(observation):
    return (observation[0], observation[1], int(observation[2]))


if __name__ == '__main__':
    """拿到环境"""
    env = gym.make("Blackjack-v1")
    env.seed(0)

    print("观察空间 = {}".format(env.observation_space))
    print("动作空间 = {}".format(env.action_space))
    print("动作数量 = {}".format(env.action_space.n))

    observation = env.reset()
    print("Initial observation: {}".format(observation))

    """
    # 测试
    while True:
        action = eval(input("Your acton: "))
        observation, reward, done, _ = env.step(int(action))
        print("observation: {}\n reward: {}\n done: {}".format(observation, reward, done))
        if done:
            observation = env.reset()
            print("New episode observation: {}".format(observation))
            continue
    """

    """
    初始化策略
    axis0: 玩家当前牌面点数总和
    axis1: 庄家公开版面点数
    axis2: 是否将玩家的1张A视为11
    axis3: 动作，分为 要牌 和 不要
    """
    policy = np.zeros((22, 11, 2, 2))
    policy[20:, :, :, 0] = 1
    policy[20:, :, :, 1] = 1
    behavior_policy = np.ones_like(policy) * 0.5


    def evaluate_monte_carlo_importance_sample(env, policy, behavior_policy, episode_nums=500000):
        q = np.zeros_like(policy)
        c = np.zeros_like(policy)

        for _ in range(episode_nums):
            state_actions = []
            observation = env.reset()

            while True:
                state = ob2state(observation)
                action = np.random.choice(env.action_space.n, p=behavior_policy[state])
                state_actions.append((state, action))
                observation, reward, done, _ = env.step(action)

                if done:
                    break

            g = reward
            rho = 1.
            for state, action in reversed(state_actions):
                """
                典型的异策学习算法，重要性采样比率
                目标策略是policy
                行为策略是behavior_policy
                行为策略指的是在实际运作过程中使用的策略，该策略生成了动作轨迹
                而目标策略则是要学习的策略
                
                每个回合之后，需要对c更新，不过这次是增加权重，而不是增加1
                对价值函数更新，更新需要乘上重要性采样比率
                以及更新重要性采样比率
                给定St和At的条件下，重要性采样比较为t+1:T-1时刻policy/behavior_policy 的连乘积
                
                所以采用逆序的方式更新该比率更为合理
                
                最后增加了一个检查机制，如果rho为0，通常是因为policy为0导致的，后面就没有更新的必要了
                因为继续更新，会出现价值函数都变成了0，不符合条件
                """
                c[state][action] += rho
                q[state][action] += (rho * (g - q[state][action]) / c[state][action])

                rho *= (policy[state][action] / behavior_policy[state][action])
                if rho == 0:
                    break
        return q


    def plot(data):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        titles = ["without ace", "with ace"]
        have_aces = [0, 1]
        extent = [12, 22, 1, 11]
        for title, have_ace, axis in zip(titles, have_aces, axes):
            dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
            axis.imshow(dat, extent=extent, origin='lower')
            axis.set_xlabel('player sum')
            axis.set_ylabel('dealer showing')
            axis.set_title(title)
        fig.show()


    q = evaluate_monte_carlo_importance_sample(env, policy, behavior_policy)
    v = (q * policy).sum(axis=-1)
    plot(v)
