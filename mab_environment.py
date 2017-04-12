import environment as e
import numpy as np


class MultiArmedBanditEnvironment(e.Environment):
    def __init__(self, k_arms, hyper_a, hyper_b):
        e.Environment.__init__(self)
        self.state_representation = [1]
        self.all_states = ["1"]
        self.action_definition = {"1": range(k_arms)}
        self.reward_model = np.random.beta(hyper_a, hyper_b, k_arms)

    def get_reward(self, state, action, next_state):
        reward = np.random.binomial(n=1, p=self.reward_model[action], size=None)
        return reward

    def determine_next_state(self, state, action):
        return state