from gym import Env, spaces
from gym.utils import seeding
from math import sin, cos, sqrt
import numpy as np


class CatchEnv(Env):
    """
        Description:
            The agent is started to the right hand side of the environment. For any given
            state the agent may choose to move to the left, right or stay still
        Source:
            n/a
        Observation:
            Type: Box(2)
            Num    Observation               Min            Max
            0      Ball Position             0              10
            1      Ball Velocity              -0.07          0.07
        Actions:
            Type: Discrete(3)
            Num    Action
            0      Move to the Left
            1      Don't move
            2      Move to the Right
        Reward:
             TBD
        Starting State:
             The starting velocity of the ball is assigned a uniform random value in
             [5 - 10].
             The starting position of the ball is always assigned to 0, 0.
        Episode Termination:
             Ball y position = 0
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CatchEnv, self).__init__()
        # env params
        self.min_x = 0
        self.max_x = 10
        self.min_y = 0
        self.max_y = 100
        self.step_count = 0

        self.goal_position = self.min_y
        self.gravity = 9.81

        self.low = np.array(
            [self.min_x, self.min_y], dtype=np.float32
        )
        self.high = np.array(
            [self.max_x, self.max_y], dtype=np.float32
        )

        self.viewer = None

        # Define action and observation space
        self.action_space = spaces.Discrete(n=3) # move left, stay still, move right
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> (dict, float, bool, dict):
        """
        Executes one timestep within the environment. Agent receives no reward until the ball hits the ground, when they
        will receive a reward equal to the inverse squared distance to the ball's landing point.
        :param action: (Int) Integer between 0 and 2 representing move left, stay still, move right respectively
        :return: next_state: dict, reward: float, is_terminal: bool, debug_info: dict
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.step_count += 1

        ball_position, ball_velocity, agent_position = self.state

        # move agent according to action
        agent_position += action - 1

        # update ball position and velocity
        ball_position_x = self.v_0 * cos(self.theta) * self.step_count * 0.01
        ball_position_y = (self.v_0 * sin(self.theta) * self.step_count * 0.01) - (0.5 * self.gravity * (self.step_count * 0.01)**2)

        # ball_position_x = np.clip(ball_position_x, self.min_x, self.max_x)
        ball_position_y = np.clip(ball_position_y, self.min_y, self.max_y)

        ball_velocity = sqrt(ball_position_x**2 + ball_position_y**2) * 1 / self.step_count

        done = ball_position_y == self.min_y

        reward = -1.0

        self.state = ((ball_position_x, ball_position_y), ball_velocity, agent_position)

        return np.array(self.state), reward, done, {}

    def reset(self):
        # state: (ball_position, ball_velocity, agent_position)
        self.step_count = 0

        # initialise agent x position
        agent_position = self.np_random.uniform(low=5, high=10)

        # initialise ball position
        ball_position = (0.0, 0.0)

        # random launch angle (theta) in [10, 80] degrees and init velocity to ensure ball lands in [5, 10]
        self.theta = self.np_random.uniform(low=0.174533, high=1.39626)
        landing_point = self.np_random.uniform(low=5, high=10)
        # v = sqrt(range*g/sin(2*theta))
        self.v_0 = sqrt((landing_point * self.gravity) / sin(2*self.theta))

        self.state = np.array([ball_position, self.v_0, agent_position])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        screen_width = 600
        screen_height = 400

        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        scale_x = screen_width / world_width
        # scale_y = screen_height / world_height

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            ball = rendering.make_circle(3)
            ball.set_color(.5, .5, .5)
            self.ballTrans = rendering.Transform()
            ball.add_attr(self.ballTrans)
            self.viewer.add_geom(ball)

            agent = rendering.FilledPolygon([(0, 0), (2.5, 7), (5, 0)])
            self.agentTrans = rendering.Transform()
            agent.add_attr(self.agentTrans)
            self.viewer.add_geom(agent)

            agent_head = rendering.make_circle(2.5)
            agent_head.set_color(.5, .5, .5)
            agent_head.add_attr(rendering.Transform(translation=(2.5, 7)))
            agent_head.add_attr(self.agentTrans)
            self.viewer.add_geom(agent_head)

        ball_pos = self.state[0]
        self.ballTrans.set_translation(
            newx=ball_pos[0] * scale_x,
            newy=ball_pos[1] * scale_x
        )

        agent_pos = self.state[2]
        self.agentTrans.set_translation(
            newx=agent_pos * scale_x,
            newy=0
        )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = CatchEnv()
    tot_reward = 0
    max_x = -100
    for i in range(50):
        state = env.reset()
        cnt = 0
        while True:
            cnt += 1
            env.render()
            # always stay still
            next_state, reward, done, info = env.step(1)
            print("Next State: {}".format(next_state))
            # print("Reward: {}".format(reward))
            # print("Done?: {}".format(done))
            if done:
                print("----------------Looped for: {}-------------------".format(cnt))
                break
    env.close()
