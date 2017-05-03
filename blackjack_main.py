import blackjack_environment as bje
import blackjack_e_greedy_agent as bjega
import numpy as np
import datetime
import value_iteration as vi
import policy_analysis as pa

env = bje.BlackjackEnvironment()
hands_to_play = 100000
#agent_greedy = bjega.BlackjackEGreedyAgent(env)
agent_greedy = bjega.BlackjackEGreedyAgent(env, epsilon_decay=1, decay=False)

show = False


def determine_action(state):
    return np.random.randint(0,2,1)

start_time = datetime.datetime.now()
for i in range(hands_to_play):
    if i % 5000 == 0:
        print("Hand Number:%d" % (i+1))
    prev_state = env.initial_state()
    reward = None
    while not prev_state[3]:  # Boolean for terminal state
        if show:
            print(prev_state)
        action = agent_greedy.determine_action(tuple(prev_state), env.possible_actions(prev_state))  # To be replaced by agent decision
        next_state = env.determine_next_state(prev_state, action)
        reward = env.get_reward(prev_state, action, next_state)
        agent_greedy.update(tuple(prev_state), action, tuple(next_state), reward)  # Update agent with prev_state, action, next_state, reward tuple
        prev_state = next_state
    if show:
        print(next_state)
    agent_greedy.episodes += 1
    if reward == 1:
        if show:
            print("HAND WON")
        agent_greedy.hands_won += 1
finish_time = datetime.datetime.now()
agent_greedy.time_taken = (finish_time - start_time).total_seconds()
print("Hands Played:%d" % agent_greedy.episodes)
print("Hands Won:%d" % agent_greedy.hands_won)
print("Time Taken:%.4f" % agent_greedy.time_taken)
print("Percent Hit:%.4f" % np.mean(agent_greedy.actions_taken))
values, policy = vi.value_iteration(env.generate_all_states(), agent_greedy.transition_beliefs, agent_greedy.reward_beliefs)
pa.basic_output(values, policy)
