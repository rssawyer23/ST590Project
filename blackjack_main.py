import blackjack_environment as bje
import blackjack_e_greedy_agent as bjega
import blackjack_optimal_agent as bjoa
import blackjack_bayesian_agent as bjba
import numpy as np
import datetime
import value_iteration as vi
import policy_analysis as pa
import matplotlib.pyplot as plt



show = False


def determine_action(state):
    return np.random.randint(0,2,1)

def run_blackjack_agent(agent, env, hands_to_play=10000):
    start_time = datetime.datetime.now()
    for i in range(hands_to_play):
        if i % 5000 == 0:
            print("Hand Number:%d" % i)
        prev_state = env.initial_state()
        reward = None
        while not prev_state[3]:  # Boolean for terminal state
            if show:
                print(prev_state)
            action = agent.determine_action(tuple(prev_state), env.possible_actions(prev_state))  # To be replaced by agent decision
            next_state = env.determine_next_state(prev_state, action)
            reward = env.get_reward(prev_state, action, next_state)
            agent.update(tuple(prev_state), action, tuple(next_state), reward)  # Update agent with prev_state, action, next_state, reward tuple
            prev_state = next_state
        if show:
            print(next_state)
        agent.episodes += 1
        agent.hands_won.append(reward == 1)
    finish_time = datetime.datetime.now()
    agent.time_taken = (finish_time - start_time).total_seconds()
    values, policy = vi.value_iteration(env.generate_all_states(), agent.transition_beliefs, agent.reward_beliefs)
    pa.policy_visualization(values, policy)


env = bje.BlackjackEnvironment()
hands = 100000
agent_egreedy = bjega.BlackjackEGreedyAgent(env)
agent_egreedy_100 = bjega.BlackjackEGreedyAgent(env, epsilon_decay=100, decay=True)
agent_random = bjega.BlackjackEGreedyAgent(env, epsilon_decay=1, decay=False)
agent_bayesian = bjba.BayesianBlackjackAgent(env)
agent_optimal = bjoa.OptimalBlackjackAgent(env)


agents = [agent_bayesian, agent_egreedy, agent_egreedy_100, agent_random, agent_optimal]
for agent in agents:
    run_blackjack_agent(agent=agent, env=env, hands_to_play=hands)
    agent.print_diagnostics(0)
    plt.plot(np.cumsum(agent.hands_won), label=agent.to_string())

plt.xlabel("Iteration")
plt.ylabel("Cumulative Reward Earned")
plt.title("Agent Reward Comparison")
plt.legend()
plt.show()
