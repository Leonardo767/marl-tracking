import numpy as np
from types import MethodType
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


def build_circle_rails(center, radius, velocity):
    vel_map = {
        1: 1000,
        2: 500,
        3: 250,
        4: 125,
    }
    velocity = velocity % len(vel_map.keys()) + 1  # prevents key errors
    theta = np.linspace(0, 2 * np.pi, num=vel_map[velocity], endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    rails = np.asarray([x, y]).T
    return rails


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.75])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            radius = np.random.uniform(-2, +2)
            center = np.asarray(
                [landmark.state.p_pos[0] - radius, landmark.state.p_pos[1]])
            vel_level = np.random.randint(1, 5)
            rails = build_circle_rails(center, radius, vel_level)
            landmark.rails = rails

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos -
                                 world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
