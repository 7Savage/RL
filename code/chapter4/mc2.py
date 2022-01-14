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


    def monte_carlo_with_exploring_start(env, episode_num=500000):
        """
        带随机初始探索的monte carlo算法
        这里得到的动作价值函数为最优价值的近似值

        研究人员发现，如果只是简单地将回合更新策略评估算法移植为
        同策回合更新算法， 时常会困于局部最优而找不到全局最优策略，
        为了解决这一问题，研究人员提出了起始探索（exploring start）
        的概念。起始探索是为了让所有可能 的状态动作对都成为可能的回合起始点，
        这样就不会遗漏任何状态动作对。

        目前，在理论上并不清楚带起始探索的同策回合更新算法是否总能够收敛到最优策略
        :param env:
        :param episode_num:
        :return:
        """
        policy = np.zeros((22, 11, 2, 2))
        # 要牌的概率100%
        policy[:, :, :, 1] = 1

        q = np.zeros_like(policy)
        c = np.zeros_like(policy)

        for _ in range(episode_num):
            """新的episode"""
            # 探索，产生新的随机状态
            state = (np.random.randint(12, 22),
                     np.random.randint(1, 11),
                     np.random.randint(2))
            action = np.random.randint(2)

            env.reset()

            """
            探索
            根据随机产生的状态，
            将玩家的牌指定任一状态即可
            """
            if state[2]:
                # 将玩家的A视为11
                env.player = [1, state[0] - 11]
            else:
                # 视为1
                if state[0] == 21:
                    env.player = [10, 9, 2]
                else:
                    env.player = [10, state[0] - 10]

            """指定庄家的可见的牌面"""
            env.dealer[0] = state[1]

            state_actions = []

            while True:
                """进入下一步"""
                state_actions.append((state, action))
                observation, reward, done, _ = env.step(action)
                if done:
                    break
                state = ob2state(observation)
                action = np.random.choice(env.action_space.n, p=policy[state])

            g = reward
            for state, action, in state_actions:
                c[state][action] += 1
                q[state][action] += (g - q[state][action]) / c[state][action]
                """
                策略改进， 同策
                这样就有希望找到最优策略
                """
                a = q[state].argmax()
                policy[state] = 0.
                policy[state][a] = 1.

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


    policy, q = monte_carlo_with_exploring_start(env)
    # 最优价值函数，取最大的一个路径即为最优状态价值
    v = q.max(axis=-1)
    plot(policy.argmax(-1))
    plot(v)


    def test_policy(env, policy, episode_nums=10000):
        result = np.zeros((episode_nums, 1), dtype=int)
        average100_result = np.zeros((int(episode_nums / 100)), dtype=float)
        for episode in range(episode_nums):
            observation = env.reset()
            while True:
                state = ob2state(observation)
                action = np.random.choice(env.action_space.n, p=policy[state])
                observation, reward, done, _ = env.step(action)
                if done:
                    break
            if reward > 0:
                result[episode] = 1
            if (episode + 1) % 100 == 0:
                average100_result[int(episode / 100)] = np.sum(result[episode - 99:episode + 1]) / 100
        return result, average100_result


    result1, result2 = test_policy(env, policy)


    def plot2(data):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        titles = ["Outcome", "averaged winning rate"]
        x_labels = ["episode", "100 episodes"]
        y_labels = ["player winning", "winning rate"]
        for title, x_label, y_label, axis, data_i in zip(titles, x_labels, y_labels, axes, data):
            axis.plot(data_i)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(title)
        fig.show()


    plot2([result1, result2])
