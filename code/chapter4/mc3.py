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


    def monte_carlo_with_soft(env, episode_nums=1000000, epsilon=0.1):
        """
        柔性策略不像前面两种回合更新算法——采用确定性策略，
        它给非最优策略了一种可选的概率,设定其为epsilon/|A(s)|概率，
        而最优动作的概率为1-epsilon + (epsilon/|A(s)|)

        这样不用初始探索也有机率访问到所有可能的状态和状态动作对。
        :param env:
        :param episode_nums:
        :param epsilon:
        :return:
        """
        policy = np.ones((22, 11, 2, 2)) * 0.5
        q = np.zeros_like(policy)
        c = np.zeros_like(policy)

        for _ in range(episode_nums):
            state_actions = []
            observation = env.reset()

            while True:
                state = ob2state(observation)
                action = np.random.choice(env.action_space.n, p=policy[state])
                state_actions.append((state, action))
                observation, reward, done, _ = env.step(action)
                if done:
                    break

            g = reward
            for state, action in state_actions:
                c[state][action] += 1.
                q[state][action] += (g - q[state][action]) / c[state][action]

                # 更新策略为柔性策略
                a = q[state].argmax()
                policy[state] = epsilon / 2.
                policy[state][a] += (1 - epsilon)
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


    policy, q = monte_carlo_with_soft(env)
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
