import agent as a
import numpy as np


class BayesianBlackjackAgent(a.Agent):
    def __init__(self, env):
        a.Agent.__init__(self, env)
        self.episodes = 0
        self.hands_won = []
        # Can initialize transition beliefs with prior here

    def determine_action(self, state, possible_actions):
        best_action = -1
        best_action_value = -10
        for action in possible_actions:
            action_value = 0
            # sample transition probability from transition beliefs (dirichlet)
            # may only want to do this once in awhile to reduce computation time
            transition_count_array = [self.transition_beliefs[state][action][next_state][0] for next_state in self.transition_beliefs[state][action].keys()]
            if np.sum(transition_count_array) == 0:
                action_taken = np.random.randint(2)
                self.actions_taken.append(action_taken)
                return action_taken

            sampled_probabilities = np.random.dirichlet(transition_count_array)

            # use sampled transition probabilities to determine action with max expected value
            for next_state, next_state_prob in zip(self.transition_beliefs[state][action].keys(), sampled_probabilities):
                if next_state_prob > 0:
                    reward_alpha = np.sum(np.array(self.reward_beliefs[state][action][next_state]) == 1)
                    reward_beta = np.sum(np.array(self.reward_beliefs[state][action][next_state]) == 0)
                    if reward_alpha > 0 and reward_beta > 0:
                        next_state_reward = np.random.beta(reward_alpha, reward_beta)
                        action_value += next_state_prob * next_state_reward
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action
        if best_action_value == -10:
            best_action = np.random.randint(2)
        self.actions_taken.append(best_action)
        return best_action

    def print_diagnostics(self, best_action):
        print(self.to_string())
        print("Hands Played:%d" % self.episodes)
        print("Percent Hands Won:%.4f" % np.mean(self.hands_won))
        print("Time Taken:%.4f" % self.time_taken)
        print("Percent Hit:%.4f" % np.mean(self.actions_taken))

    def to_string(self):
        return "Bayesian Agent"