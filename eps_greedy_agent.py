import agent as a

import numpy as np


class EpsGreedyAgent(a.Agent):
    # Each arm (action) has a reward belief that is its probability of giving reward 1
    def __init__(self, k_arms, epsilon_decay = 50):
        a.Agent.__init__(self)
        self.n = k_arms
        self.counts = [0] * k_arms
        self.values = [0.] * k_arms
        self.decay = epsilon_decay

    # Reward will be 0 or 1, this updates the posterior
    # State is the currently selected arm????
    def update_reward_beliefs(self, state, action, next_state, reward):
        self.counts[action] = self.counts[action] + 1
        n = self.counts[action]
        value = self.values[action]
        # Update the reward ratio for this arm
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[action] = new_value

    def determine_action(self, state, possible_actions):
        # Sample reward probabilities from reward_beliefs
        # Select max action (index) of sampled_reward_beliefs
        epsilon = self.get_epsilon()
        if np.random.random() > epsilon:
            # Greedy path, select best arm
            return np.argmax(self.values)
        else:
            # Exporation path, select random arm
            return np.random.randint(self.n)

    def get_epsilon(self):
        total = np.sum(self.counts)
        return float(self.decay)/(total + float(self.decay))