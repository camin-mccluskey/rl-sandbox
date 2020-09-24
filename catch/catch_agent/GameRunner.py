from catch.catch_gym.catch_gym.envs import CatchEnv


def run(runs=10):
    env = CatchEnv()
    for i in range(runs):
        state = env.reset()
        while True:
            env.render()
            # chose action
            ball_x_pos = state[0][0]
            agent_x_pos = state[2]
            # stay still
            action = 1
            if ball_x_pos < agent_x_pos:
                action = 0
            elif ball_x_pos > agent_x_pos:
                action = 2
            state, reward, done, info = env.step(action=action)
            if done:
                break
    env.close()


if __name__ == '__main__':
    run()
