import blackjack_environment as bje
import numpy as np

env = bje.BlackjackEnvironment()
prev_state = env.initial_state()
print(prev_state)
action = 1
reward = None
cont = True

def determine_action(state):
    return np.random.randint(0,2,1)

while cont:
    action = determine_action(prev_state)  # To be replaced by agent decision
    next_state = env.determine_next_state(prev_state, action)
    reward, cont = env.get_reward(prev_state, action, next_state)
    prev_state = next_state
    # Update agent with prev_state, action, next_state, reward tuple
    print(next_state)
print(reward)
