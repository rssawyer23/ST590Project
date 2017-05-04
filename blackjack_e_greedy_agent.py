import agent as a
import numpy as np

class BlackjackEGreedyAgent(a.Agent):
    def __init__(self, env, epsilon_decay = 50, decay=True, gamma=0.9):
        a.Agent.__init__(self, env)
        self.decay = epsilon_decay
        self.should_decay = decay
        self.final_epsilon = -1.
        self.exploring = True
        self.episodes = 0
        self.hands_won = []
        self.gamma = gamma

    def determine_action(self, state, possible_actions):
        epsilon = self.get_epsilon()
        action_taken = -1
        if self.exploring:
            self.final_epsilon = epsilon
            if np.random.random() > epsilon:
                action_taken = self.greedy_action(state, possible_actions)
            else:
                action_taken = np.random.randint(2)
        else:
            action_taken = self.greedy_action(state, possible_actions)
        self.actions_taken.append(action_taken)
        return action_taken

    def greedy_action(self, state, possible_actions):
        best_action = -1
        best_action_value = -10
        for action in possible_actions:
            action_value = 0
            for next_state in self.transition_beliefs[state][action].keys():
                if self.transition_beliefs[state][action][next_state][0] > 0:
                    next_state_probability = self.transition_beliefs[state][action][next_state][0] / \
                                             self.transition_beliefs[state][action][next_state][1]
                    next_state_reward = np.mean(self.reward_beliefs[state][action][next_state])
                    action_value += next_state_probability * next_state_reward
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        if best_action_value == -10:
            best_action = np.random.randint(2)
        return best_action

    def get_epsilon(self):
        if self.should_decay:
            return float(self.decay)/(self.episodes + float(self.decay))
        else:
            return self.decay

    def print_diagnostics(self, best_action):
        print(self.to_string())
        print("Hands Played:%d" % self.episodes)
        print("Percent Hands Won:%.4f" % np.mean(self.hands_won))
        print("Time Taken:%.4f" % self.time_taken)
        print("Percent Hit:%.4f" % np.mean(self.actions_taken))

    def to_string(self):
        if self.should_decay:
            return "Epsilon Greedy Agent Decay=%d" % self.decay
        else:
            return "Epsilon Greedy Agent Decay=%s" % self.should_decay

