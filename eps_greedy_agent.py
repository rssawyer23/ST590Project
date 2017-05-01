import agent as a
import numpy as np


class EpsGreedyAgent(a.Agent):
    # Each arm (action) has a reward belief that is its probability of giving reward 1
    def __init__(self, k_arms, epsilon_decay = 50):
        a.Agent.__init__(self)
        self.arms = k_arms
        self.counts = [0] * k_arms
        self.reward_beliefs = [0.] * k_arms
        self.decay = epsilon_decay

    # Reward will be 0 or 1, this updates the posterior
    # State is the currently selected arm????
    def update_reward_beliefs(self, state, action, next_state, reward):
        self.counts[action] = self.counts[action] + 1
        n = self.counts[action]
        value = self.reward_beliefs[action]
        # Update the reward ratio for this arm
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.reward_beliefs[action] = new_value

    def determine_action(self, state, possible_actions):
        # Sample reward probabilities from reward_beliefs
        # Select max action (index) of sampled_reward_beliefs
        epsilon = self.get_epsilon()
        if np.random.random() > epsilon:
            # Greedy path, select best arm
            return np.argmax(self.reward_beliefs)
        else:
            # Exploration path, select random arm
            return np.random.randint(self.arms)

    def get_epsilon(self):
        total = np.sum(self.counts)
        return float(self.decay)/(total + float(self.decay))

    def print_diagnostics(self):
        print("Epsilon Greedy Agent")
        print("\tAverage Reward: %.4f" % np.mean(self.rewards_earned))
        print("\tTime Taken %.4f" % self.time_taken)
        print("\tGreedy Agent Beliefs")
        for arm in range(self.arms):
            print("\t\tArm %d: %.4f" % (arm+1, self.reward_beliefs[arm]))

    def to_string(self):
        return "Epsilon Greedy Agent"