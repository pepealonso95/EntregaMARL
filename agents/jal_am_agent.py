import numpy as np
from base.game import SimultaneousGame, AgentID
from base.agent import Agent

class JointActionLearningAgentModellingAgent(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID, alpha=0.1, gamma=0.99, epsilon=0.1, min_epsilon=0.01, epsilon_decay=0.995, seed=None, print_every_n_updates=200): # Added print_every_n_updates
        super().__init__(game=game, agent=agent)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.num_actions = self.game.num_actions(self.agent)
        # Assume two agents for simplicity
        self.other_agent = [a for a in self.game.agents if a != self.agent][0]
        self.num_other_actions = self.game.num_actions(self.other_agent)
        self.Q = np.ones((self.num_actions, self.num_other_actions)) * 0.1  # Empieza optimista para explorar
        self.joint_count = np.ones((self.num_actions, self.num_other_actions))  # For initial smoothing
        self.opp_count = np.ones(self.num_other_actions)  # For opponent policy estimation
        self.last_action = None
        self.last_opp_action = None
        self.last_reward = None
        self.last_obs = None
        self.last_joint_action = None  # To explicitly track the last joint action
        self.opp_policy_estimate = None # Added for logging
        self.expected_Q_values = None # Added for logging
        self.early_returns = 0 # Counter for early returns in update()
        self.update_count = 0  # Initialize update counter
        self.learn = True  # Initialize learning flag
        self.print_every_n_updates = print_every_n_updates # Store print frequency

    def reset(self):
        # Reset all tracking variables
        self.last_action = None
        self.last_opp_action = None
        self.last_reward = None
        self.last_obs = None
        self.last_joint_action = None
        self.opp_policy_estimate = None # Reset logged value
        self.expected_Q_values = None # Reset logged value
        # Reset early returns counter (for debugging)
        self.early_returns = 0
        self.update_count = 0 # Reset update counter

    def store_joint_action(self, joint_action):
        """Store the joint action performed in the environment.
        This method is kept for backward compatibility but is no longer used.
        Actions are now stored directly in play_episode() before the environment step.
        """
        print(f"DEBUG: Agent {self.agent} store_joint_action() - This method is deprecated. Actions are now set directly.")
        self.last_joint_action = joint_action

    def _get_joint_action(self):
        """Return the last actions taken by this agent and the opponent.
        This method is now simplified to just return the stored values,
        which are set directly in play_episode() before the environment step.
        """
        # print(f"DEBUG: Agent {self.agent} _get_joint_action() - Using stored actions: my={self.last_action}, opp={self.last_opp_action}")
        return self.last_action, self.last_opp_action

    def _estimate_opp_policy(self):
        # Empirical distribution over opponent's actions
        return self.opp_count / np.sum(self.opp_count)

    def action(self):
        # Estimate expected Q for each action using current opponent policy
        opp_policy = self._estimate_opp_policy()
        expected_Q = np.dot(self.Q, opp_policy)

        # Store for logging
        self.opp_policy_estimate = opp_policy
        self.expected_Q_values = expected_Q

        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
            # print(f"DEBUG: Agent {self.agent} action() - Random action: {action}")
        else:
            max_indices = np.where(expected_Q == np.max(expected_Q))[0]
            action = np.random.choice(max_indices)
            # print(f"DEBUG: Agent {self.agent} action() - Greedy action: {action}, max indices: {max_indices}")
            
        # Note: We don't store the action in self.last_action here anymore
        # It's now handled directly in play_episode() to ensure joint action coherence
        
        # print(f"DEBUG: Agent {self.agent} action() - Q stats: mean={np.mean(self.Q):.6f}, max={np.max(self.Q):.6f}, min={np.min(self.Q):.6f}")
        
        return int(action)

    def update(self, reward: float):
        """
        Updates the Q-table and opponent model based on the last transition.
        """
        if not self.learn or self.last_action is None or self.last_opp_action is None:
            return

        self.update_count += 1 # Increment update counter

        # Convert actions to integer indices if they are not already
        my_action_idx = int(self.last_action)
        opp_action_idx = int(self.last_opp_action)

        # Increment counts for opponent modeling
        self.joint_count[my_action_idx, opp_action_idx] += 1
        self.opp_count[opp_action_idx] += 1
        
        # Update opponent policy estimate
        # This should be done before calculating expected_Q_values for the current state
        # as it uses the updated counts.
        for opp_a in range(self.num_other_actions):
            if self.opp_count[opp_a] > 0:
                self.opp_policy_estimate[opp_a] = self.opp_count[opp_a] / np.sum(self.opp_count)
            else:
                self.opp_policy_estimate[opp_a] = 1.0 / self.num_other_actions # Uniform if no counts

        # Calculate the expected Q-value for the next state (which is the current state as it's stateless)
        # This involves summing over opponent's possible next actions, weighted by their estimated policy
        
        # First, calculate Q-values for all our possible next actions, assuming current opponent policy
        next_expected_Q_values_for_my_actions = np.zeros(self.num_actions)
        for next_my_a in range(self.num_actions):
            sum_q_opp_policy = 0
            for next_opp_a in range(self.num_other_actions):
                sum_q_opp_policy += self.Q[next_my_a, next_opp_a] * self.opp_policy_estimate[next_opp_a]
            next_expected_Q_values_for_my_actions[next_my_a] = sum_q_opp_policy
        
        # The max of these is the value of the next state (V(s'))
        max_next_expected_Q = np.max(next_expected_Q_values_for_my_actions)

        # Q-learning update rule
        current_q_value = self.Q[my_action_idx, opp_action_idx]
        td_target = reward + self.gamma * max_next_expected_Q
        td_error = td_target - current_q_value
        
        if self.update_count % self.print_every_n_updates == 0: # Conditional printing
            # Log details before Q-value update
            print(f"Agent {self.agent} updating (Update #{self.update_count}): MyAction={my_action_idx}, OppAction={opp_action_idx}, Reward={reward:.2f}")
            print(f"  Q_before={current_q_value:.4f}, TD_target={td_target:.4f}, TD_error={td_error:.4f}")

        self.Q[my_action_idx, opp_action_idx] += self.alpha * td_error

        if self.update_count % self.print_every_n_updates == 0: # Conditional printing
            # Log Q-value after update
            print(f"  Q_after={self.Q[my_action_idx, opp_action_idx]:.4f}")

        # Epsilon decay (moved from training loop to here, if desired for per-update decay)
        # if self.epsilon > self.min_epsilon:
        # self.epsilon *= self.epsilon_decay_rate
        # Ensure epsilon does not go below min_epsilon
        # self.epsilon = max(self.min_epsilon, self.epsilon)

    def policy(self):
        # Return current expected policy (greedy w.r.t. expected Q)
        opp_policy = self._estimate_opp_policy()
        expected_Q = np.dot(self.Q, opp_policy)
        policy = np.zeros(self.num_actions)
        best_action = np.argmax(expected_Q)
        policy[best_action] = 1.0
        return policy