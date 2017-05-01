import mab_environment as mabe
import eps_greedy_agent as epsag
import mab_bayesian_agent as mba
import datetime
import numpy as np
import matplotlib.pyplot as plt

#Main file for running Agent/Environment interactions

# General Thompson Sampling Algorithm
# initialize
# for t in range(timesteps):
#   agent.determine_action
#   environment.get_reward
#   agent.update
# results


def run_mab_agent(agent, env, time_steps=10):
    start_time = datetime.datetime.now()
    print("Starting agent %s at %s" % (agent.to_string(), start_time))
    for t in range(time_steps):
        action = agent.determine_action(possible_actions=None, state=None)
        next_state = env.determine_next_state(state=1,action=action)
        reward = env.get_reward(action=action, state=1, next_state=next_state)
        agent.update(state=1, action=action, next_state=next_state, reward=reward)
    finish_time = datetime.datetime.now()
    time_taken = (finish_time - start_time).total_seconds()
    agent.time_taken = time_taken


k_arms = 5
time_steps = 10
state = 1  # could use random selection here
tot_reward = 0
eps_decay = 20
show = True

env = mabe.MultiArmedBanditEnvironment(k_arms=k_arms, hyper_a=0.3, hyper_b=0.5)
eps = epsag.EpsGreedyAgent(k_arms=k_arms, epsilon_decay=eps_decay)
bay = mba.MabBayesianAgent(k_arms=k_arms, prior_a=0.5, prior_b=0.5)
env.environment_diagnostics()

agents = [eps, bay]
for agent in agents:
    run_mab_agent(agent=agent, env=env, time_steps=time_steps)
    agent.print_diagnostics()
    if show:
        plt.plot(np.cumsum(agent.rewards_earned), label=agent.to_string())

if show:
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Reward Earned")
    plt.title("Agent Reward Comparison")
    plt.legend()
    plt.show()


# eps_start = datetime.datetime.now()
# for t in range(time_steps):
#     action = eps.determine_action(possible_actions=None, state=None)
#     reward = env.get_reward(action=action, state=None, next_state=None)
#     next_state = env.determine_next_state(state=1, action=action)
#     eps.update(state=1, action=action, next_state=next_state, reward=reward)
# eps_finish = datetime.datetime.now()
# eps_time = (eps_finish - eps_start).total_seconds()
# eps_average_reward = np.mean(eps.rewards_earned)


