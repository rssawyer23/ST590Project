import mab_environment as mabe
#Main file for running Agent/Environment interactions

# General Thompson Sampling Algorithm
# initialize
# for t in range(timesteps):
#   agent.determine_action
#   environment.get_reward
#   agent.update
# results

env = mabe.MultiArmedBanditEnvironment(k_arms=5, hyper_a=0.5, hyper_b=0.5)
print env.reward_model
