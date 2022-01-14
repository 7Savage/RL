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
    
    当玩家牌面大于等于20时，则不再要牌
    当玩家牌面小于20时，则继续要牌
    
    这样的一个策略为确定性策略
    """
    policy = np.zeros((22, 11, 2, 2))
    # >=20 不要的概率为100%
    policy[20:, :, :, 0] = 1
    # <20 要
    policy[:20, :, :, 1] = 1


    def evaluate_action_monte_carlo(env, policy, episode_num=500000):
        """
        该函数用monte carlo算法评估该确定性策略

        :param env:
        :param policy:
        :param episode_num:
        :return:
        """
        # 初始化动作价值函数
        q = np.zeros_like(policy)
        # 初始化计数器
        c = np.zeros_like(policy)
        for _ in range(episode_num):
            """新的回合开始"""
            # 每个episode, 状态动作序列则更新
            state_actions = []
            # 初始化环境
            observation = env.reset()

            while True:
                """开始下一步"""
                # 把观察到的状态做一点小改动
                state = ob2state(observation)
                # 根据策略选择动作
                action = np.random.choice(env.action_space.n, p=policy[state])
                # 将状态动作对保存
                state_actions.append((state, action))
                # 执行当前动作，并观察结果
                observation, reward, done, _ = env.step(action)
                # 如果done为True, 则此次episode结束
                if done:
                    break
            # 每个回合有一个reward,胜出为1，失败为-1， 其余情况的reward为0
            """
            # TODO
            由于这里只有在回合结束之后都有reward，所以就不需要考虑正着或反着计算，
            而且gamma为1
            """
            g = reward
            for state, action in state_actions:
                # 统计每个状态动作对
                """
                由于每次要牌，都会使得牌面点数变化，因此在一个回合中，
                每个状态只会出现一次，所以在这个案例里面，无所谓first-visit
                与every-visit
                
                一个回合结束，则将状态动作对出现的次数统计到计数器中，
                
                first-visit: 每个回合如果出现则更新计数器，出现多次则只更新计数器1次
                every-visit: 只要该状态-动作出现就更新计数器，
                这是二者的唯一区别
                """
                c[state][action] += 1
                # 状态动作价值
                q[state][action] += (g - q[state][action]) / c[state][action]

        return q


    """
    q 为动作价值函数 四个维度
    v 为状态价值函数 三个维度
    * 表示对应元素相乘， sum(axis=-1)指按照最后一个维度相加，即
    """
    q = evaluate_action_monte_carlo(env, policy)
    v = (q * policy).sum(axis=-1)


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


    plot(v)
