import numpy as np


class ReplayBuffer(object):
    def __init__(self):
        self.buffer = []
        self.trajectory = []

    def add_to_buffer(self, data):
        if len(self.buffer) > 1e+6:
            print('clearing replay buffer...')
            self.buffer = self.buffer[int(1e+4):len(self.buffer)]
        self.buffer.append(data)

    def sample_batch(self,batch_length):
        if batch_length == 0:
            randomized_int_list = np.arange(len(self.buffer))
            loop_length =len(self.buffer)
        else:
            randomized_int_list = np.random.randint(0, len(self.buffer), size=batch_length)
            loop_length = batch_length
        batch_transitions = []
        for i in range(loop_length):
            batch_transitions.append(self.buffer[randomized_int_list[i]])
        B = list(zip(*batch_transitions))
        return B

    def add_unprocessed_transition(self, state, next_state, action, reward, done, goal, a_goal):
        self.trajectory.append((state, action, reward, next_state, done, goal, a_goal))

    def clear_episodic_transition(self):
        self.trajectory = []

    def process_samples(self, process_normal_transitions,perform_her):
        traj = list(zip(*self.trajectory))
        states = traj[0]
        actions = traj[1]
        rewards = traj[2]
        next_states = traj[3]
        dones = traj[4]
        goals = traj[5]
        a_goals = traj[6]
        her_traj_length = len(self.trajectory)
        if perform_her:
            for t in range(her_traj_length):
                K = 4
                for k in range(K):
                    future_actual_goal = np.random.randint(t, her_traj_length)

                    if future_actual_goal == t:
                        reward = 0
                        done_num = 1
                    else:
                        done_num = 0
                        reward = -1

                    self.add_to_buffer((states[t], next_states[t], actions[t],
                                        reward, done_num, a_goals[future_actual_goal]))

        if process_normal_transitions:
            for t in range(len(self.trajectory)):
                    self.add_to_buffer(
                        (states[t], next_states[t], actions[t], rewards[t], dones[t], goals[t]))
        self.clear_episodic_transition()


    def clear_transitions(self,amount=None):
        if amount == None:
            self.buffer = []
        else:
            self.buffer = self.buffer[int(amount):len(self.buffer)]
        pass