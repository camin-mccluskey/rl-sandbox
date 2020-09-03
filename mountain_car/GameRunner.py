from mountain_car.Model import Model
from mountain_car.Memory import Memory
from mountain_car.config import config

import random
import math
import gym

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps, decay, discount, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._discount = discount
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0
        max_x = -100
        while True:
            if self._render:
                self._env.render()

            action = self._choose_action(state)
            next_state, reward, done, info = self._env.step(action)
            if next_state[0] >= -0.25:
                reward += 100
            elif next_state[0] >= 0.25:
                reward += 200
            elif next_state[0] >= 0.6:
                reward += 1000

            # reward = next_state position value (linear reward)
            # reward = next_state[0]

            # no penalty for moving backwards
            # reward = max(0, next_state[0])

            # to record best achieved position
            if next_state[0] > max_x:
                max_x = next_state[0]

            # is the game complete? If so, set the next state to
            # None for storage sake
            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the epsilon value
            self._steps += 1
            self._eps = self._min_eps + (self._max_eps - self._min_eps) * math.exp(-self._decay * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)
                break

        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model._num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model._batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model._num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self._discount * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)


if __name__ == "__main__":
    # construct OpenAI Gym
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)

    # infer action and state space from env
    num_states = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.n

    model = Model(num_states, num_actions, config['BATCH_SIZE'])
    mem = Memory(50000)

    with tf.compat.v1.Session() as sess:
        sess.run(model._var_init)
        gr = GameRunner(sess, model, env, mem, config['MAX_EPSILON'], config['MIN_EPSILON'],
                        config['LAMBDA'], config['GAMMA'])
        num_episodes = 500
        cnt = 0
        while cnt < num_episodes:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
            gr.run()
            cnt += 1
        plt.plot(gr._reward_store)
        plt.show()
        plt.close("all")
        plt.plot(gr._max_x_store)
        plt.show()
