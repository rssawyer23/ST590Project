import agent as a
import numpy as np

class OptimalBlackjackAgent(a.Agent):
    def __init__(self, env):
        a.Agent.__init__(self, env)
        self.episodes = 0
        self.hands_won = []

    def determine_action(self, state, possible_actions):  # Manual encoding of optimal policy deduced from random exploration over 1 million hands
        if state[1]:
            if state[0] <= 17:
                action = 1
            elif state[0] >= 19:
                action = 0
            elif state[2] >= 8:
                action = 1
            else:
                action = 0
        else:
            if state[0] == 11:
                action = 1
            elif state[0] <= 14 and state[2] >= 7:
                action = 1
            elif state[2] == 11 and state[0] <= 17:
                action = 1
            else:
                action = 0
        self.actions_taken.append(action)
        return action

    def print_diagnostics(self, best_action):
        print(self.to_string())
        print("Hands Played:%d" % self.episodes)
        print("Percent Hands Won:%.4f" % np.mean(self.hands_won))
        print("Time Taken:%.4f" % self.time_taken)
        print("Percent Hit:%.4f" % np.mean(self.actions_taken))

    def to_string(self):
        return "Optimal Agent Policy"
