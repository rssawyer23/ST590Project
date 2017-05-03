
class Agent:
    '''
        Class for a simple Reinforcement Learning agent that can interact with the environment
        Rewards Earned - list of rewards earned at each time stamp (interaction with environment)
        Transition Beliefs - probability the agent believes of next state given state and action
        Reward Beliefs - reward the agent believes of state, action pair (Q-Values)
    '''
    def __init__(self, env):
        self.rewards_earned = []
        self.actions_taken = []
        self.transition_beliefs = None
        self.reward_beliefs = None
        self.time_taken = None
        self.converged_iteration = -1
        self.initialize_beliefs(env)

    def initialize_beliefs(self, env):
        self.transition_beliefs = {}
        self.reward_beliefs = {}
        for curr_state in env.generate_all_states():
            self.transition_beliefs[curr_state] = {}
            self.reward_beliefs[curr_state] = {}
            for action_type in env.possible_actions(curr_state):
                self.transition_beliefs[curr_state][action_type] = {}
                self.reward_beliefs[curr_state][action_type] = {}
                for next_state in env.generate_all_states():
                    self.transition_beliefs[curr_state][action_type][next_state] = [0, 0]
                    self.reward_beliefs[curr_state][action_type][next_state] = []

    # Update the transition model, assuming arguments passed as tuples
    def update_transitions(self, state, action, next_state):
        self.transition_beliefs[state][action][next_state][0] += 1

        for potential_next_state in self.transition_beliefs[state][action].keys():
            self.transition_beliefs[state][action][potential_next_state][1] += 1

    # Update the reward model, assuming arguments passed as tuples
    def update_reward_beliefs(self, state, action, next_state, reward):
        self.reward_beliefs[state][action][next_state].append(reward)

    # After taking an action, updating the agent's beliefs about the environment
    def update(self, state, action, next_state, reward):
        self.rewards_earned.append(reward)
        self.update_transitions(state, action, next_state)
        self.update_reward_beliefs(state, action, next_state, reward)

    # Action selection policy of agent using its current state and possible actions to determine what action to take
    # Essentially this will wrap over a policy, which maps states -> actions, to include exploit/explore dynamic
    def determine_action(self, state, possible_actions):
        pass

    def print_diagnostics(self, best_action):
        print("Reward Beliefs %s" % self.reward_beliefs)

    def to_string(self):
        return "Base Agent"