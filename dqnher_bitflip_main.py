import torch
torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_num_threads(1)
import numpy as np
from bitflip_env import BitFlippingEnv
from goal_based_dqn import GoalBased_DeepQNetwork
import pickle

z = 25
#READ ME#

#In the bitflipping environment the agent starts at a random state represented by an array
#of zero or one bit variables e.g [0,1,0,1,1,1,1,0]. Note the array is of length z.
#In the same starting state the agent is also given a random goal state e.g [1,1,1,1,1,1,1,1]
#In each step in the environment the agent can flip one bit from zero to one or, one to zero.
#The agents gets exactly z steps to go from the given starting state to the given goal state
# and is only given a reward of zero if it arrives at the goal state, -1 otherwise.
#Inorder to solve the environment for a z >= 20/25, HER is needed (currently afaik).
#Last time I checked it can solve the environment for z = 50,
# might require some hyperparameter tuning from it's current setup.

env = BitFlippingEnv(z)
state_dim = z
goal_dim = z
action_dim = z

episodes_per_eval = 16
total_evals = 300

policy = GoalBased_DeepQNetwork(state_dim, goal_dim, action_dim)

data_list = []
for eval in range(total_evals):
        successes_per_eval = 0
        for e in range(episodes_per_eval):
            state, goal = env.reset()
            for t in range(z):
                if np.random.uniform(0, 10) < 3 and eval < 50:
                    action = np.random.randint(0, z)
                else:
                    action = policy.get_policy_action(np.array(state), np.array(goal))
                next_state, reward, done = env.step(action)
                onehot = np.zeros(action_dim)
                onehot[action] = 1
                action = onehot

                done_bool = 0
                if done:
                    successes_per_eval += 1
                    done_bool = 1

                policy.replay_buffer.add_unprocessed_transition(
                    state, next_state, action, reward, done_bool, goal, next_state)
                state = next_state
                if done:
                    break
            policy.replay_buffer.process_samples(True, True)
        print('eval:', eval)
        print('success rate: ', successes_per_eval / episodes_per_eval)
        data_list.append(successes_per_eval / episodes_per_eval)
        policy.update(40)

