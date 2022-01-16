import gym
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    env = env.unwrapped
    print('观察空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('位置范围 = {}'.format((env.min_position, env.max_position)))
    print('速度范围 = {}'.format((-env.max_speed, env.max_speed)))
    print('目标位置 = {}'.format(env.goal_position))


    def test():
        """
        测试总是向右
        """
        positions, velocities = [], []
        observation = env.reset()
        max_steps = 200
        while max_steps:
            max_steps -= 1
            positions.append(observation[0])
            velocities.append(observation[1])

            next_observation, reward, done, _ = env.step(2)
            if done:
                break
            observation = next_observation
        if next_observation[0] > 0.5:
            print("成功到达")
        else:
            print("失败退出")

        fig, ax = plt.subplots()
        ax.plot(positions, label="position")
        ax.plot(velocities, label="velocity")
        ax.legend()
        fig.show()


    test()
