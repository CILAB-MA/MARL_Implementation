import gym
import rware


def call_env():
    env = gym.make("rware-tiny-2ag-v1")
    obs = env.reset()
    actions = env.action_space.sample()
    done = [False, False]

    while not done[0]:
        print(f'actions: {actions}')
        n_obs, reward, done, info = env.step(actions)

        print(f'done: {done}')  # [False, False]
        print(f'reward: {reward}')  # [0.0, 0.0]
    env.close()


if __name__ == '__main__':
    call_env()