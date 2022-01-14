import numpy as np

np.random.seed(0)
import pandas as pd
import gym

if __name__ == '__main__':
    space_name = ['观测空间', '动作空间', '奖励范围', '最大步数']
    df = pd.DataFrame(columns=space_name)

    env_specs = gym.envs.registry.all()
    for env_spec in env_specs:
        env_id = env_spec.id
        try:
            env = gym.make(env_id)
            observation_space = env.observation_space
            action_space = env.action_space
            reward_range = env.reward_range
            max_episode_steps = None
            if isinstance(env, gym.wrappers.time_limit.TimeLimit):
                max_episode_steps = env._max_episode_steps
            df.loc[env_id] = [observation_space, action_space, reward_range, max_episode_steps]
        except:
            pass
    with pd.option_context('display.max_rows', None):
        print(df)
# if __name__ == '__main__':
    # 查看当前gym库已经注册了哪些环境
    from gym import envs

    env_specs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in env_specs]
    # print(env_ids)
    # 得到环境对象
    env = gym.make('CartPole-v0')

    # 初始化环境对象，调用完后，返回智能体的初始观测，是np.array对象
    env.reset()

    # 使用对象的核心是使用环境对象的step()方法
    # 该方法接收一个动作参数，然后返回四个参数
    """
    四个参数分别是
    观测 observation:np.array对象
    奖励 reward: float类型
    回合结束标志 done: bool类型， gym 库中的实验环境大多是回合制的
    其它信息 info:dict类型
    """
    # 该函数的参数取自动作空间
    # 使用如下方法获取
    action = env.action_space.sample()
    # 每次调用完step()之后，只会前进一步，因此需要将该函数放在循环结构中

    # 在reset()或step()之后，可以使用以下语句来图形化显示当前环境
    # ！注意，使用了图形界面接口的话，关闭的最好方式是使用env.close()
    # 而不是直接关闭界面，这样可能会导致内存无法释放，甚至会导致死机
    env.render()

    # 使用完环境后，需要使用下列语句将环境关闭,
    env.close()
