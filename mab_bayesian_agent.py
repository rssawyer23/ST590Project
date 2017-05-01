import agent as a
import numpy as np


class MabBayesianAgent(a.Agent):
    # Each arm (action) has a reward belief that is its probability of giving reward 1
    # Each of these arms is modeled by a Beta-Binomial Posterior distribution
    # The Beta prior is specified by the arguments to the initialization here
    def __init__(self, k_arms, prior_a, prior_b):
        a.Agent.__init__(self)
        self.reward_beliefs = dict()
        self.arms = k_arms
        for k in range(k_arms):
            self.reward_beliefs[k] = [prior_a, prior_b]

    # Reward will be 0 or 1, this updates the posterior
    # The first element of the reward belief for an arm is the posterior alpha, the second is the posterior beta
    def update_reward_beliefs(self, state, action, next_state, reward):
        self.reward_beliefs[action][0] += reward
        self.reward_beliefs[action][1] += 1 - reward

    def determine_action(self, state, possible_actions):
        # Sample reward probabilities from reward_beliefs
        # Select max action (index) of sampled_reward_beliefs
        sampled_probabilities = []
        for i in range(self.arms):
            p_i = float(np.random.beta(self.reward_beliefs[i][0],self.reward_beliefs[i][1], size=1))
            sampled_probabilities.append(p_i)
        return np.argmax(sampled_probabilities)

    def print_diagnostics(self):
        print("Bayesian Agent")
        print("\tAverage Reward: %.4f" % np.mean(self.rewards_earned))
        print("\tTime Taken: %.4f" % self.time_taken)
        print("\tReward Beliefs")
        for arm in range(self.arms):
            arm_p = self.reward_beliefs[arm][0] / (self.reward_beliefs[arm][0] + self.reward_beliefs[arm][1])
            print("\t\tArm %d: %.4f" % (arm+1, arm_p))

    def to_string(self):
        return "Bayesian Agent"