import numpy as np

def value_iteration(all_states, transition_beliefs, reward_beliefs, gamma=0.9):
    converged = False
    value_dict = dict()
    policy = dict()
    for state in all_states:
        if state[0] > 21:
            value_dict[state] = -1.
        else:
            value_dict[state] = 0.
        policy[state] = -1.
    iteration = 0

    while not converged:
        iteration += 1
        delta = 0.
        for state in all_states:
            value = value_dict[state]
            best_action = -1
            best_action_value = -10000
            for action in transition_beliefs[state].keys():
                potential_new_value = 0.
                experienced = False
                for next_state in transition_beliefs[state][action].keys():
                    if transition_beliefs[state][action][next_state][0] > 0:
                        experienced = True
                        trans_prob = transition_beliefs[state][action][next_state][0] / transition_beliefs[state][action][next_state][1]
                        potential_new_value += trans_prob * (np.mean(reward_beliefs[state][action][next_state]) + gamma * value_dict[next_state])
                if potential_new_value > best_action_value:
                    if experienced:
                        best_action = action
                    best_action_value = potential_new_value
            if abs(best_action_value - value) > delta:
                delta = abs(best_action_value - value)
            value_dict[state] = best_action_value
            policy[state] = best_action
        if delta < 0.01:
            converged = True
    print("Iterations: %d" % iteration)
    return value_dict, policy