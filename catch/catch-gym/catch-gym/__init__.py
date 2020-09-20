from gym.envs.registration import register

register(
    id='catch-v0',
    entry_point='catch-gym.envs:CatchEnv',
)