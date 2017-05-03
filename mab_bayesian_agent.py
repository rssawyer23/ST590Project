import agent as a
import numpy as np


class MabBayesianAgent(a.Agent):
    # Each arm (action) has a reward belief that is its probability of giving reward 1
    # Each of these arms is modeled by a Beta-Binomial Posterior distribution
    # The Beta prior is specified by the arguments to the initialization here
    def __init__(self, env, k_arms, prior_a, prior_b):
        a.Agent.__init__(self, env)
        self.reward_beliefs = dict()
        self.arms = k_arms
        self.exploring = True
        self.max_arm = None
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
        action_taken = -1
        if self.exploring:
            sampled_probabilities = []
            for i in range(self.arms):
                p_i = float(np.random.beta(self.reward_beliefs[i][0],self.reward_beliefs[i][1], size=1))
                sampled_probabilities.append(p_i)
            if len(self.rewards_earned) > 300 and len(self.rewards_earned) % 50 == 0:
                opt_prob, opt_arm = self.optimal_belief()
                if opt_prob > 0.95:
                    self.exploring = False
                    self.max_arm = opt_arm
                    self.converged_iteration = len(self.rewards_earned)
            action_taken = np.argmax(sampled_probabilities)
        else:
            action_taken = self.max_arm
        self.actions_taken.append(action_taken)
        return action_taken

    # Function for calculating posterior probability that arm with highest expected reward is maximum reward arm
    # This is done by sampling each arms posterior distribution and calculating the number of samples where max arm is max sample
    def optimal_belief(self, sample_size=10000):
        samples = []
        max_arm_exp = 0
        max_arm_ind = None
        for i in range(self.arms):
            arm_exp = self.reward_beliefs[i][0] / float(self.reward_beliefs[i][1] + self.reward_beliefs[i][0])
            if arm_exp > max_arm_exp:
                max_arm_exp = arm_exp
                max_arm_ind = i
            p_i = np.random.beta(self.reward_beliefs[i][0],self.reward_beliefs[i][1],size=sample_size)
            samples.append(p_i)
        samples = np.array(samples)
        max_ind_array = np.apply_along_axis(np.argmax, arr=samples, axis=0)
        return np.mean(max_ind_array == max_arm_ind), max_arm_ind

    def print_diagnostics(self, best_action):
        print("Bayesian Agent")
        print("\tAverage Reward: %.4f (%.4f)" % (np.mean(self.rewards_earned), np.std(self.rewards_earned)))
        print("\tPercent Correct Arm: %.4f" % np.mean(np.array(self.actions_taken) == best_action))
        print("\tTime Taken: %.4f" % self.time_taken)
        print("\tMax Arm Belief: %.4f, Arm %d" % self.optimal_belief())
        print("\tConverged Iteration: %d" % self.converged_iteration)
        print("\tReward Beliefs")
        for arm in range(self.arms):
            arm_p = self.reward_beliefs[arm][0] / (self.reward_beliefs[arm][0] + self.reward_beliefs[arm][1])
            print("\t\tArm %d: %.4f" % (arm+1, arm_p))

    def to_string(self):
        return "Bayesian Agent"