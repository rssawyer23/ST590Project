#Python file for defining an MDP environment and dynamics
class Environment:
    '''
        Class for defining the dynamics of the environment, need to be specified by specific problem
        State Representation - The possible states of the environment, represented by a vector of state variables
        Action Definition - The possible actions from each state, is a state:action set mapping
        Transition Probabilities - A transition matrix from S to S' for each action in action definition
        Initial Probabilities - A state:probability mapping states to their probability of being the start state
        Reward Model - Given a (state, action, next state) triplet, outputs a scalar reward
    '''
    def __init__(self):
        self.state_representation = None
        self.all_states = None
        self.action_definition = None
        self.transition_probabilities = None
        self.initial_probabilities = None
        self.reward_model = None

    # Following functions to illustrate the desired functionality of the environment
    def generate_all_states(self):
        pass

    def possible_actions(self, state):
        return self.action_definition[state]

    def transition_probability(self, state, action, next_state):
        return self.transition_probabilities[action][state, next_state]

    def get_reward(self, state, action, next_state):
        pass

    def start_probability(self, state):
        return self.initial_probabilities[state]

    # Used for stochastic environment interaction
    # Agent gives current state and action taken
    # Environment uses this information to stochastically determine which next state the agent will be in
    def determine_next_state(self, state, action):
        pass
