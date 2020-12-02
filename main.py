from particle_env.make_env import make_env
from multiagent.policy import InteractivePolicy
import time

env = make_env('tracking')
env.render()

# create interactive policies for each agent
policies = [InteractivePolicy(env, i) for i in range(env.n)]
# execution loop
obs_n = env.reset()
while True:
    # query for action from each agent's policy
    act_n = []
    for i, policy in enumerate(policies):
        act_n.append(policy.action(obs_n[i]))
    # step environment
    obs_n, reward_n, done_n, _ = env.step(act_n)
    # render all agent views
    env.render()
    # display rewards
    # for agent in env.world.agents:
    #     print(agent.name + " reward: %0.3f" % env._get_reward(agent))
