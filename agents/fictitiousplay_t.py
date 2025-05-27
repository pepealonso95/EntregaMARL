from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        if seed is not None:
            np.random.seed(seed=seed)
        
        self.count: dict[AgentID, ndarray] = {}
        for a in game.agents:
            action_space = game.action_spaces[a]
            num_actions = action_space.n
            if initial is not None and a in initial and \
               isinstance(initial[a], (list, np.ndarray)) and len(initial[a]) == num_actions:
                self.count[a] = np.array(initial[a], dtype=float)
            else:
                self.count[a] = np.ones(num_actions, dtype=float) 

        self.learned_policy: dict[AgentID, ndarray] = {}
        for a in game.agents:
            sum_counts = np.sum(self.count[a])
            if sum_counts == 0:
                num_actions = game.action_spaces[a].n
                self.learned_policy[a] = np.ones(num_actions, dtype=float) / num_actions if num_actions > 0 else np.array([], dtype=float)
            else:
                self.learned_policy[a] = self.count[a] / sum_counts

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        
        for joint_action in product(*agents_actions):
            action_dict = {agent: action for agent, action in zip(g.agents, joint_action)}
            g.set_actions(action_dict)
            rewards[tuple(joint_action)] = g.reward(self.agent)
            
        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        
        try:
            agent_game_index = self.game.agents.index(self.agent)
        except ValueError:
            return utility 

        for joint_action_tuple, reward_for_agent in rewards.items():
            my_action_in_joint_action = joint_action_tuple[agent_game_index]
            
            prob_others_actions = 1.0
            valid_joint_action = True
            for i, other_agent_id in enumerate(self.game.agents):
                if other_agent_id != self.agent:
                    other_agent_action_in_joint_action = joint_action_tuple[i]
                    if other_agent_id in self.learned_policy and \
                       len(self.learned_policy[other_agent_id]) > other_agent_action_in_joint_action:
                        prob_others_actions *= self.learned_policy[other_agent_id][other_agent_action_in_joint_action]
                    else:
                        valid_joint_action = False
                        break 
            
            if valid_joint_action:
                utility[my_action_in_joint_action] += prob_others_actions * reward_for_agent
            
        return utility
    
    def bestresponse(self):
        utility = self.get_utility()
        num_actions = self.game.action_spaces[self.agent].n
        
        if num_actions == 0:
            return 0

        if np.all(np.isclose(utility, utility[0] if len(utility) > 0 else 0.0)) or len(utility) == 0:
            return np.random.choice(num_actions)

        max_utility = np.max(utility)
        max_indices = np.where(np.isclose(utility, max_utility))[0]
        
        if len(max_indices) == 0: 
            return np.random.choice(num_actions)

        a = np.random.choice(max_indices)
        return int(a)
     
    def update(self) -> None:
        last_joint_action_obs = self.game.observe(self.agent)

        if last_joint_action_obs is None or not isinstance(last_joint_action_obs, dict):
            return

        for other_agent_id in self.game.agents:
            if other_agent_id == self.agent:
                continue

            if other_agent_id in last_joint_action_obs:
                other_agent_last_action = last_joint_action_obs[other_agent_id]
                
                if isinstance(other_agent_last_action, (int, np.integer)) and \
                   0 <= other_agent_last_action < len(self.count[other_agent_id]):
                    self.count[other_agent_id][other_agent_last_action] += 1.0
                    
                    sum_counts = np.sum(self.count[other_agent_id])
                    num_other_actions = self.game.action_spaces[other_agent_id].n
                    if sum_counts == 0: 
                        self.learned_policy[other_agent_id] = np.ones(num_other_actions, dtype=float) / num_other_actions if num_other_actions > 0 else np.array([], dtype=float)
                    else:
                        self.learned_policy[other_agent_id] = self.count[other_agent_id] / sum_counts

    def action(self):
        return self.bestresponse()