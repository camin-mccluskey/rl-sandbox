from gym import Env, spaces
from gym.utils import seeding
from math import sin, cos, sqrt
from typing import Tuple
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
            0      Ball x Position           0              10
            1      Ball y Position           14.2           0
            2      Ball Velocity             0              +inf
            3      Agent x Position          0              10
        Actions:
            Type: Discrete(3)
            Num    Action
            0      Move to the Left
            1      Don't move
            2      Move to the Right
        Reward:
            TBD
        Starting State:
            The starting position of the agent is assigned a uniform random value in [5, 10].
            The starting position of the ball is always assigned to 0, 0.
            The launch angle (theta) of the ball is assigned a uniform random value in [10, 80] degrees.
            The landing point, along x axis, is pre assigned a uniform random value in [5, 10]
            The initial velocity of the ball is such that for chosen theta ball will land at chosen landing point
        Episode Termination:
            Ball y Position = 0
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CatchEnv, self).__init__()
        # env params
        self._min_launch_angle = 0.174533 # 10deg in rad
        self._max_launch_angle = 1.39626 # 80deg in rad
        self._step_count = 0
        self._gravity = 9.81
        self._time_dilation_factor = 0.01

        # bound environment size
        self._min_x = 0
        self._max_x = 10
        self._min_y = 0
        self._max_y = 14.2 # contains largest possible arc

        self._goal_position = self._min_y

        self._low = np.array(
            [self._min_x, self._min_y], dtype=np.float32
        )
        self._high = np.array(
            [self._max_x, self._max_y], dtype=np.float32
        )

        self._viewer = None

        # Define action and observation space
        self.action_space = spaces.Discrete(n=3) # move left, stay still, move right
        self.observation_space = spaces.Box(low=self._low, high=self._high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        """
        Executes one timestep within the environment. Agent receives no reward until the ball hits the ground, when they
        will receive a reward equal to the inverse squared distance to the ball's landing point.
        :param action: (Int) Integer between 0 and 2 representing move left, stay still, move right respectively
        :return: next_state: dict, reward: float, is_terminal: bool, debug_info: dict
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self._step_count += 1

        ball_position, ball_velocity, agent_position = self._state

        # move agent according to action
        agent_position += (action - 1) * self._time_dilation_factor * 2

        # clip agent position to bounds
        agent_position = np.clip(agent_position, self._min_x, self._max_x)

        # update ball position and velocity
        ball_position_x = self._v_0 * cos(self._theta) * self._step_count * self._time_dilation_factor
        ball_position_y = (self._v_0 * sin(self._theta) * self._step_count * self._time_dilation_factor) - \
                          (0.5 * self._gravity * (self._step_count * self._time_dilation_factor) ** 2)

        # clip ball position to bounds
        ball_position_x = np.clip(ball_position_x, self._min_x, self._max_x)
        ball_position_y = np.clip(ball_position_y, self._min_y, self._max_y)

        # velocity parallel to projectile path
        ball_velocity = sqrt(ball_position_x**2 + ball_position_y**2) * 1 / self._step_count

        done = ball_position_y == self._min_y

        reward = 0
        if done:
            reward = -abs(ball_position_x - agent_position)

        self._state = ((ball_position_x, ball_position_y), ball_velocity, agent_position)

        return np.array(self._state), reward, done, {}

    def reset(self):
        # state: (ball_position, ball_velocity, agent_position)
        self._step_count = 0

        # initialise agent x position
        agent_position = self.np_random.uniform(low=5, high=10)

        # initialise ball position
        ball_position = (0.0, 0.0)

        # random launch angle (theta) in [10, 80] degrees and init velocity to ensure ball lands in [5, 10]
        self._theta = self.np_random.uniform(low=self._min_launch_angle, high=self._max_launch_angle)
        landing_point = self.np_random.uniform(low=5, high=self._max_x)

        self._v_0 = sqrt((landing_point * self._gravity) / sin(2 * self._theta))

        self._state = np.array([ball_position, self._v_0, agent_position])
        return np.array(self._state)

    def render(self, mode='human'):
        # Render the environment to the screen
        screen_width = 600
        screen_height = 400

        world_width = self._max_x - self._min_x
        world_height = self._max_y - self._min_y
        scale_x = screen_width / world_width
        scale_y = screen_height / world_height

        if self._viewer is None:
            from gym.envs.classic_control import rendering
            self._viewer = rendering.Viewer(screen_width, screen_height)

            ball = rendering.make_circle(3)
            ball.set_color(.5, .5, .5)
            self._ballTrans = rendering.Transform()
            ball.add_attr(self._ballTrans)
            self._viewer.add_geom(ball)

            agent = rendering.FilledPolygon([(0, 0), (2.5, 7), (5, 0)])
            self._agentTrans = rendering.Transform()
            agent.add_attr(self._agentTrans)
            self._viewer.add_geom(agent)

            agent_head = rendering.make_circle(2.5)
            agent_head.set_color(.5, .5, .5)
            agent_head.add_attr(rendering.Transform(translation=(2.5, 7)))
            agent_head.add_attr(self._agentTrans)
            self._viewer.add_geom(agent_head)

        ball_pos = self._state[0]
        self._ballTrans.set_translation(
            newx=ball_pos[0] * scale_x,
            newy=ball_pos[1] * scale_y
        )

        agent_pos = self._state[2]
        self._agentTrans.set_translation(
            newx=agent_pos * scale_x,
            newy=0
        )

        return self._viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None


if __name__ == '__main__':
    env = CatchEnv()
    tot_reward = 0
    max_x = -100
    for i in range(10):
        state = env.reset()
        cnt = 0
        while True:
            cnt += 1
            env.render()
            # always stay still
            next_state, reward, done, info = env.step(1)
            print("Next State: {}".format(next_state))
            print("Reward: {}".format(reward))
            print("Done?: {}".format(done))
            if done:
                break
    env.close()
