from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


benchmark = False
scenario_name = 'simple'
# load scenario from script
scenario = scenarios.load(scenario_name + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment
if benchmark:
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, scenario.benchmark_data)
else:
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation)
