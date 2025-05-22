import copy
from pettingzoo.utils import env
from pettingzoo.utils.env import ParallelEnv

ObsDict = env.ObsDict
AgentID = env.AgentID
ActionDict = env.ActionDict

class SimultaneousGame(ParallelEnv):

    observations: ObsDict
    rewards: dict[AgentID, float]
    terminations: dict[AgentID, bool]
    truncations: dict[AgentID, bool]
    infos: dict[AgentID, dict]

    agent_name_mapping: dict[AgentID, int]

    def observation_space(self, agent: AgentID):
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID):
        return self.action_spaces[agent]

    def num_actions(self, agent: AgentID):
        return self.action_space(agent).n
    
    def action_iter(self, agent: AgentID):
        return range(self.action_space(agent).start, self.action_space(agent).n)
        
    def observe(self, agent: AgentID):
        return self.observations[agent]
    
    def reward(self, agent: AgentID):
        return self.rewards[agent]
    
    def clone(self):
        """Creates a deep copy of the game with proper state reset."""
        game = copy.deepcopy(self)
        game.reset()
        return game
    
    def set_actions(self, actions: ActionDict):
        """Sets the actions for all agents and advances the game state."""
        _, self.rewards, _, _, _ = self.step(actions)





