import numpy as np
from base.game import SimultaneousGame, AgentID
from base.agent import Agent

class JointActionLearningAgentModellingAgent(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID, alpha=0.1, gamma=0.99, epsilon=0.1, min_epsilon=0.01, epsilon_decay=0.995, seed=None):
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
        self.Q = np.zeros((self.num_actions, self.num_other_actions))
        self.joint_count = np.ones((self.num_actions, self.num_other_actions))  # For initial smoothing
        self.opp_count = np.ones(self.num_other_actions)  # For opponent policy estimation
        self.last_action = None
        self.last_opp_action = None
        self.last_reward = None
        self.last_obs = None

    def reset(self):
        self.last_action = None
        self.last_opp_action = None
        self.last_reward = None
        self.last_obs = None

    def _get_joint_action(self):
        # Try to extract the last joint action from the observation
        obs = self.game.observe(self.agent)
        if isinstance(obs, dict) and 'action' in obs:
            joint_action = obs['action']
            if isinstance(joint_action, (tuple, list)):
                my_idx = self.game.agent_name_mapping[self.agent]
                opp_idx = self.game.agent_name_mapping[self.other_agent]
                return joint_action[my_idx], joint_action[opp_idx]
        return None, None

    def _estimate_opp_policy(self):
        # Empirical distribution over opponent's actions
        return self.opp_count / np.sum(self.opp_count)

    def action(self):
        # Estimate expected Q for each action using current opponent policy
        opp_policy = self._estimate_opp_policy()
        expected_Q = np.dot(self.Q, opp_policy)
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            max_indices = np.where(expected_Q == np.max(expected_Q))[0]
            action = np.random.choice(max_indices)
        self.last_action = action
        # Save last observed opponent action for update
        _, opp_action = self._get_joint_action()
        self.last_opp_action = opp_action
        return int(action)

    def update(self):
        # Get last joint action
        my_action, opp_action = self._get_joint_action()
        if my_action is None or opp_action is None:
            return
        reward = self.game.reward(self.agent)
        # Update counts for opponent policy estimation
        self.opp_count[opp_action] += 1
        self.joint_count[my_action, opp_action] += 1
        # Q-learning update for joint action
        opp_policy = self._estimate_opp_policy()
        expected_next_Q = np.dot(self.Q, opp_policy).max()
        td_target = reward + self.gamma * expected_next_Q
        td_error = td_target - self.Q[my_action, opp_action]
        self.Q[my_action, opp_action] += self.alpha * td_error
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def policy(self):
        # Return current expected policy (greedy w.r.t. expected Q)
        opp_policy = self._estimate_opp_policy()
        expected_Q = np.dot(self.Q, opp_policy)
        policy = np.zeros(self.num_actions)
        best_action = np.argmax(expected_Q)
        policy[best_action] = 1.0
        return policy