import numpy as np

import matplotlib.pyplot as plt
import gym


def ob2state(observation):
    return observation[0], observation[1], int(observation[2])


def evaluate_action_monte_carlo(env, policy, episode_num=500000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 玩一回合
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break  # 回合结束
        g = reward  # 回报
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
    return q


def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    titles = ["without ace", "with ace"]
    have_aces = [0, 1]
    extent = [12, 22, 1, 11]
    for title, have_ace, axis, in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
        axis.imshow(dat, extent=extent, origin='lower')
        axis.set_xlabel('player sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)

    fig.show()


if __name__ == '__main__':
    env = gym.make("Blackjack-v1")
    env.seed(0)
    print('观察空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('动作数量 = {}'.format(env.action_space.n))

    policy = np.zeros((22, 11, 2, 2))
    policy[20:, :, :, 0] = 1  # >= 20 时收手
    policy[:20, :, :, 1] = 1  # < 20 时继续

    q = evaluate_action_monte_carlo(env, policy)  # 动作价值
    v = (q * policy).sum(axis=-1)  # 状态价值
    plot(v)
