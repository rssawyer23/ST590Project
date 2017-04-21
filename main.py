import mab_environment as mabe
import eps_greedy_agent as epsag

#Main file for running Agent/Environment interactions

# General Thompson Sampling Algorithm
# initialize
# for t in range(timesteps):
#   agent.determine_action
#   environment.get_reward
#   agent.update
# results

k_arms = 5;
time_steps = 1000
state = 1  # could use random selection here
tot_reward = 0
eps_decay = 20

env = mabe.MultiArmedBanditEnvironment(k_arms=k_arms, hyper_a=0.5, hyper_b=0.5)
eps = epsag.EpsGreedyAgent(k_arms=k_arms, epsilon_decay=eps_decay)

for t in range(time_steps):
    action = eps.determine_action(possible_actions=None, state=None)
    reward = env.get_reward(action=action, state=None, next_state=None)
    eps.update_reward_beliefs(action=action, reward=reward,state=None, next_state=None)
    tot_reward += reward

print (tot_reward)
