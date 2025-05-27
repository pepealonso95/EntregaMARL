import numpy as np
from typing import Dict, Optional 
from base.agent import Agent
from base.game import SimultaneousGame, ActionDict

class RegretMatching(Agent):

    def __init__(self, game: SimultaneousGame, agent: str, initial: Optional[np.ndarray]=None, seed: Optional[int]=None) -> None:
        super().__init__(game=game, agent=agent)
        num_actions = self.game.num_actions(self.agent)

        if initial is None:
          self.curr_policy = np.full(num_actions, 1.0/num_actions if num_actions > 0 else 0.0)
        else:
          self.curr_policy = initial.copy()
        
        self.cum_regrets = np.zeros(num_actions)
        self.sum_policy = self.curr_policy.copy() 
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1

        if seed is not None:
            np.random.seed(seed=seed)

    def _calculate_regrets(self, observed_joint_action: Dict[str, int]) -> np.ndarray:
        actual_utility = self.game.reward(self.agent)
        
        num_agent_actions = self.game.num_actions(self.agent)
        u_counterfactual = np.zeros(num_agent_actions, dtype=float)
        
        game_clone_for_cf = self.game.clone()
        
        for alt_action_for_self in range(num_agent_actions):
            counterfactual_joint_action = observed_joint_action.copy()
            counterfactual_joint_action[self.agent] = alt_action_for_self
            
            game_clone_for_cf.set_actions(counterfactual_joint_action)
            u_counterfactual[alt_action_for_self] = game_clone_for_cf.reward(self.agent)
            
        regrets = u_counterfactual - actual_utility
        return regrets
    
    def _update_internal_state_based_on_last_observation(self):
        if self.game.rewards[self.agent] is None:
            return

        observed_joint_action: Optional[Dict[str, int]] = None

        obs_from_game = self.game.observe(self.agent)
        if isinstance(obs_from_game, dict):
            is_potential_joint_action = True
            if not all(agent_id in obs_from_game for agent_id in self.game.agents):
                is_potential_joint_action = False
            if is_potential_joint_action:
                for agent_id in self.game.agents:
                    if not isinstance(obs_from_game.get(agent_id), int):
                        is_potential_joint_action = False
                        break
            if is_potential_joint_action:
                observed_joint_action = obs_from_game

        if observed_joint_action is None and hasattr(self.game, '_last_joint_action_input'):
            last_input = getattr(self.game, '_last_joint_action_input')
            if isinstance(last_input, dict):
                is_potential_joint_action = True
                if not all(agent_id in last_input for agent_id in self.game.agents):
                    is_potential_joint_action = False
                if is_potential_joint_action:
                    for agent_id in self.game.agents:
                         if not isinstance(last_input.get(agent_id), int):
                            is_potential_joint_action = False
                            break
                if is_potential_joint_action:
                    observed_joint_action = last_input
        
        if observed_joint_action is None:
            return

        regrets_from_last_turn = self._calculate_regrets(observed_joint_action)
        self.cum_regrets += regrets_from_last_turn

        regrets_plus = np.maximum(0, self.cum_regrets)
        regret_sum = np.sum(regrets_plus)
        
        num_actions = self.game.num_actions(self.agent)
        if regret_sum > 0:
            self.curr_policy = regrets_plus / regret_sum
        else:
            self.curr_policy = np.full(num_actions, 1.0/num_actions if num_actions > 0 else 0.0)
        
        self.sum_policy += self.curr_policy
        self.niter += 1
        if self.niter > 0:
             self.learned_policy = self.sum_policy / self.niter
        else:
             self.learned_policy = self.curr_policy.copy()


    def action(self) -> int:
        self._update_internal_state_based_on_last_observation()

        num_actions = self.game.num_actions(self.agent)
        if num_actions == 0:
             raise ValueError(f"Agent {self.agent} has no actions available.")

        current_policy_sum = np.sum(self.curr_policy)
        if not np.isclose(current_policy_sum, 1.0) or current_policy_sum <= 0: 
            self.curr_policy = np.full(num_actions, 1.0/num_actions if num_actions > 0 else 0.0)
            if num_actions == 0:
                 raise ValueError(f"Agent {self.agent} has no actions available even after policy reset.")
            if num_actions > 0 and not np.isclose(np.sum(self.curr_policy), 1.0):
                 self.curr_policy = np.full(num_actions, 1.0/num_actions)


        if not np.isclose(np.sum(self.curr_policy), 1.0) and num_actions > 0:
            self.curr_policy = self.curr_policy / np.sum(self.curr_policy)
        
        if num_actions > 0 and np.sum(self.curr_policy) > 0:
            chosen_action_array = np.random.multinomial(1, self.curr_policy, size=1)[0]
            return np.argmax(chosen_action_array).item() 
        elif num_actions > 0:
            return np.random.choice(num_actions)
        else:
            raise ValueError(f"Agent {self.agent} cannot choose an action, num_actions is {num_actions}.")

    
    def policy(self) -> np.ndarray:
        return self.learned_policy
