import gym
import numpy as np
from _dqn_extra import DeepQNetwork

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy = DeepQNetwork(state_dim, action_dim)

max_time_steps = 500
episodes = 200
data_list = []
for e in range(episodes):
        reward_sum = 0
        state = env.reset()
        for i in range(max_time_steps):
            if np.random.uniform(0, 10) < 4 and e < 50:
                action = env.action_space.sample()
            else:
                action = policy.get_policy_action(state)
            next_state, reward, done, info = env.step(action)
            onehot = np.zeros(action_dim)
            onehot[action] = 1
            action = onehot
            reward_sum += reward
            done_bool = 0
            if done: done_bool = 1
            policy.replay_buffer.add_to_buffer(
                (state, next_state, action, reward, done_bool))

            state = next_state
            if done:
                break
        if reward_sum < 490:
            policy.update(200)
        print('reward_sum: ', reward_sum)