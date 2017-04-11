import agent as a


class MabBayesianAgent(a.Agent):
    # Each arm (action) has a reward belief that is its probability of giving reward 1
    # Each of these arms is modeled by a Beta-Binomial Posterior distribution
    # The Beta prior is specified by the arguments to the initialization here
    def __init__(self, k_arms, prior_a, prior_b):
        a.Agent.__init__(self)
        self.reward_beliefs = dict()
        for k in range(k_arms):
            self.reward_beliefs[k] = [prior_a, prior_b]

    # Reward will be 0 or 1, this updates the posterior
    def update_reward_beliefs(self, state, action, next_state, reward):
        self.reward_beliefs[action][0] += reward
        self.reward_beliefs[action][1] += 1 - reward

    def determine_action(self, state, possible_actions):
        # Sample reward probabilities from reward_beliefs
        # Select max action (index) of sampled_reward_beliefs
        pass