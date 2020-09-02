# Mountain Car Environment: Reinforcement Learning

The object of this game is to get the car to go up the right-side hill to get to the flag. There’s one problem however, the car doesn’t have enough power to motor all the way up the hill. Instead, the car (agent) needs to learn that it must motor up one hill for a bit, then accelerate down the hill and back up the other side, and repeat until it builds up enough momentum to make it to the top of the hill.

The environment is represented by [this](https://github.com/openai/gym/wiki/MountainCar-v0) OpenAI Gym.

## Environment State Space

| Num | Observation | Min   | Max  |
|:----|:------------|:------|:-----|
| 0   | Position    | -1.2  | 0.6  |
| 1   | Velocity    | -0.07 | 0.07 |

*Goal: To have the car reach the flag at position 0.5*

## Action Space

| Num | Action     |
|:----|:-----------|
| 0   | Push Left  |
| 1   | No Push    |
| 2   | Push Right |


## Reward Function

<img src="/mountain_car/tex/4f4f480fba7e40c56333b7c61f21dc47.svg?invert_in_darkmode&sanitize=true" align=middle width=395.92919355pt height=67.39786349999999pt/>


