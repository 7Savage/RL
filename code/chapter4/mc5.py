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


    def monte_carlo_importance_sample(env, episode_nums=500000):
        policy = np.zeros((22, 11, 2, 2))
        # 初始化为不要牌的概率为1
        policy[:, :, :, 0] = 1
        # 行为策略为柔性策略
        behavior_policy = np.ones_like(policy) * 0.5
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
                c[state][action] += rho
                q[state][action] += (rho * (g - q[state][action]) / c[state][action])
                # 取出价值最大的动作
                a = q[state].argmax()
                policy[state] = 0.
                policy[state][a] = 1.0

                if a != action:
                    """
                    这种情况下，rho一定为0， 就没有更新的必要了
                    """
                    break
                rho /= behavior_policy[state][action]

        return policy, q


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


    policy, q = monte_carlo_importance_sample(env)
    v = q.max(axis=-1)
    plot(policy.argmax(axis=-1))
    plot(v)

