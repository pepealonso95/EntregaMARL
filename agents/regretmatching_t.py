import numpy as np
from typing import Dict, Optional 
from base.agent import Agent
from base.game import SimultaneousGame, ActionDict # Ensure ActionDict is imported if used in type hints

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
        self.niter = 1 # niter=1 because sum_policy is initialized with curr_policy

        if seed is not None:
            np.random.seed(seed=seed)

    def _calculate_regrets(self, observed_joint_action: Dict[str, int]) -> np.ndarray:
        # actual_utility is the reward the agent received for its action in observed_joint_action
        # This reward should be from the main game object, reflecting the actual outcome.
        actual_utility = self.game.reward(self.agent)
        
        num_agent_actions = self.game.num_actions(self.agent)
        u_counterfactual = np.zeros(num_agent_actions, dtype=float)
        
        # Create a fresh game clone for each counterfactual evaluation.
        # IMPORTANT: self.game.clone() as defined in base/game.py creates a deepcopy AND RESETS the game.
        # This is suitable for stateless games (RPS, MP, Blotto) where reward depends only on current actions.
        # For stateful games (e.g., Foraging), this will calculate counterfactuals from an initial state,
        # not the current game state, which is incorrect for proper regret calculation in those contexts.
        game_clone_for_cf = self.game.clone()
        
        for alt_action_for_self in range(num_agent_actions):
            # Create the counterfactual joint action: self takes alt_action, others take their observed actions
            counterfactual_joint_action = observed_joint_action.copy()
            counterfactual_joint_action[self.agent] = alt_action_for_self
            
            # Simulate this counterfactual joint action in the cloned (and reset) game
            # For stateless games, set_actions (which calls step) will give the correct one-shot utility.
            game_clone_for_cf.set_actions(counterfactual_joint_action)
            u_counterfactual[alt_action_for_self] = game_clone_for_cf.reward(self.agent)
            
        regrets = u_counterfactual - actual_utility
        return regrets
    
    def _update_internal_state_based_on_last_observation(self):
        # This method is called at the beginning of action(), so it uses observations from the *previous* turn (t-1)
        # to update regrets and policy for choosing an action at turn t.

        # If self.game.rewards[self.agent] is None, it means no step has been taken yet after a reset in this episode.
        if self.game.rewards[self.agent] is None:
            return # Cannot update regrets if no prior action/reward

        observed_joint_action: Optional[Dict[str, int]] = None

        # Try to get joint action from the game's standard observation method.
        # For RPS, MP, Blotto, self.game.observe(self.agent) returns the joint action dict.
        obs_from_game = self.game.observe(self.agent)
        if isinstance(obs_from_game, dict):
            # Check if it looks like a joint action dictionary (keys are agent_ids, values are ints)
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

        # Fallback for games that might use a custom attribute (e.g., Foraging, as hinted by notebook debugs)
        # This relies on the test environment or game wrapper to populate _last_joint_action_input.
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
            # If we couldn't determine the last joint action, we can't calculate regrets.
            # This might happen if the game doesn't provide it in a recognized way.
            # print(f"Warning: Agent {self.agent} could not determine the last joint action for regret update. Skipping update.")
            return

        # We have a valid observation from the previous turn.
        regrets_from_last_turn = self._calculate_regrets(observed_joint_action)
        self.cum_regrets += regrets_from_last_turn

        # Update current strategy (curr_policy) based on new cumulative regrets
        regrets_plus = np.maximum(0, self.cum_regrets)
        regret_sum = np.sum(regrets_plus)
        
        num_actions = self.game.num_actions(self.agent)
        if regret_sum > 0:
            self.curr_policy = regrets_plus / regret_sum
        else:
            # Default to uniform random policy if all regrets are non-positive
            self.curr_policy = np.full(num_actions, 1.0/num_actions if num_actions > 0 else 0.0)
        
        self.sum_policy += self.curr_policy
        self.niter += 1
        # The learned policy is the average strategy over iterations
        if self.niter > 0 : # self.niter starts at 1, so this should always be true after first init
             self.learned_policy = self.sum_policy / self.niter
        else: # Should not happen if niter starts at 1
             self.learned_policy = self.curr_policy.copy()


    def action(self) -> int:
        self._update_internal_state_based_on_last_observation()

        num_actions = self.game.num_actions(self.agent)
        if num_actions == 0:
             raise ValueError(f"Agent {self.agent} has no actions available.")

        current_policy_sum = np.sum(self.curr_policy)
        # Check if policy is valid, otherwise reset to uniform (and log/warn)
        if not np.isclose(current_policy_sum, 1.0) or current_policy_sum <= 0: 
            # print(f"Warning: Agent {self.agent} has invalid current policy (sum={current_policy_sum}). Resetting to uniform.")
            self.curr_policy = np.full(num_actions, 1.0/num_actions if num_actions > 0 else 0.0)
            # If num_actions was 0 and policy was reset, re-check num_actions to avoid division by zero if it's still 0
            if num_actions == 0:
                 raise ValueError(f"Agent {self.agent} has no actions available even after policy reset.")
            # Ensure policy sum is 1 after reset if num_actions > 0
            if num_actions > 0 and not np.isclose(np.sum(self.curr_policy), 1.0):
                 # This case should ideally not be reached if reset is correct
                 self.curr_policy = np.full(num_actions, 1.0/num_actions)


        # Choose action based on current strategy (not the learned/average one for exploration)
        # Ensure curr_policy sums to 1 for multinomial
        if not np.isclose(np.sum(self.curr_policy), 1.0) and num_actions > 0 :
            self.curr_policy = self.curr_policy / np.sum(self.curr_policy) # Normalize if slightly off due to float precision
        
        # np.random.multinomial requires probabilities to sum to 1.
        # Handle cases where curr_policy might be all zeros (e.g. if num_actions is 0 initially, though guarded)
        if num_actions > 0 and np.sum(self.curr_policy) > 0:
            chosen_action_array = np.random.multinomial(1, self.curr_policy, size=1)[0]
            return np.argmax(chosen_action_array).item() 
        elif num_actions > 0: # Fallback to uniform random if policy is zero sum but actions exist
            # print(f"Warning: Agent {self.agent} curr_policy sums to zero. Choosing uniformly random action.")
            return np.random.choice(num_actions)
        else: # Should be caught by num_actions == 0 check earlier
            raise ValueError(f"Agent {self.agent} cannot choose an action, num_actions is {num_actions}.")

    
    def policy(self) -> np.ndarray:
        return self.learned_policy
