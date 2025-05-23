import numpy as np
from base.game import SimultaneousGame, AgentID
from base.agent import Agent
import math

class IQLAgentConfig:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, min_epsilon=0.01, epsilon_decay=0.995, 
                 max_t=1000, seed=None, optimistic_init=0.1, use_reward_shaping=True):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_t = max_t
        self.seed = seed
        self.optimistic_init = optimistic_init
        self.use_reward_shaping = use_reward_shaping

class IQLAgent(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID, config: IQLAgentConfig = None):
        super().__init__(game=game, agent=agent)
        self.alpha = getattr(config, 'alpha', 0.1) if config is not None else 0.1
        self.gamma = getattr(config, 'gamma', 0.99) if config is not None else 0.99
        self.epsilon = getattr(config, 'epsilon', 0.9) if config is not None else 0.9  # Higher initial exploration
        self.min_epsilon = getattr(config, 'min_epsilon', 0.01) if config is not None else 0.01
        self.epsilon_decay = getattr(config, 'epsilon_decay', 0.9999) if config is not None else 0.9999  # Slower decay
        self.optimistic_init = getattr(config, 'optimistic_init', 0.1) if config is not None else 0.1
        self.use_reward_shaping = getattr(config, 'use_reward_shaping', True) if config is not None else True
        self.learn = True
        self.seed = getattr(config, 'seed', None) if config is not None else None
        if self.seed is not None:
            np.random.seed(self.seed)
        self.num_actions = self.game.num_actions(self.agent)
        self.Q = {}  # Q-table: state (tuple) -> np.array of action values
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_obs = None  # Store previous observation for reward shaping

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def _obs_to_state(self, obs):
        """
        Extract essential features from the observation to create a simplified state.
        This helps reduce the state space complexity for better learning.
        """
        if isinstance(obs, np.ndarray):
            # For Foraging environment, extract only key features
            if len(obs.shape) == 3:  # Typical shape for grid observations
                # Extract agent position (look for '1' in player channel)
                agent_pos = None
                # Try to identify agent position (usually in first channel) and food positions
                if obs.shape[0] >= 2:  # If we have at least 2 channels
                    # Find agent position (usually marked with 1 in first channel)
                    agent_channel = obs[0]
                    agent_pos = np.unravel_index(agent_channel.argmax(), agent_channel.shape)
                    
                    # Find food positions (usually in second channel)
                    food_channel = obs[1] if obs.shape[0] > 1 else None
                    food_pos = None
                    if food_channel is not None and food_channel.max() > 0:
                        # Get position of nearest food
                        food_positions = np.transpose(np.nonzero(food_channel))
                        if len(food_positions) > 0:
                            if agent_pos:
                                # Find nearest food
                                distances = [np.sqrt((f[0] - agent_pos[0])**2 + (f[1] - agent_pos[1])**2) for f in food_positions]
                                nearest_idx = np.argmin(distances)
                                food_pos = tuple(food_positions[nearest_idx])
                            else:
                                # Just take the first food
                                food_pos = tuple(food_positions[0])
                
                # Create a simplified state representation
                if agent_pos and food_pos:
                    # Return relative position to the nearest food
                    return (food_pos[0] - agent_pos[0], food_pos[1] - agent_pos[1])
                elif agent_pos:
                    # No food found, just return agent position
                    return agent_pos
            
            # Fallback if we couldn't extract features: use truncated version of the flattened array
            flat_obs = obs.flatten()
            if len(flat_obs) > 10:  # If observation is large, take only key elements
                return tuple(flat_obs[:10])  # Use first 10 elements
            return tuple(flat_obs)
        elif isinstance(obs, (list, tuple)):
            return tuple(np.array(obs).flatten())
        elif isinstance(obs, dict):  # Add this condition to handle dict observations
            # Convert dict to a sorted tuple of items to make it hashable
            return tuple(sorted(obs.items()))
        else:
            # Handles other types, like single primitive values
            return tuple([obs])

    def action(self):
        obs = self.game.observe(self.agent)
        # Store observation for reward shaping
        self.last_obs = obs
        
        state = self._obs_to_state(obs)
        
        # Optimistic initialization to encourage exploration
        if state not in self.Q:
            self.Q[state] = np.ones(self.num_actions) * self.optimistic_init
        
        # Improved exploration strategy with annealed epsilon-greedy
        if self.learn and np.random.rand() < self.epsilon:
            # Use smarter exploration: bias toward potentially useful actions
            # Randomly select action, but give higher probability to non-NONE actions
            if self.num_actions > 1:  # If we have more than just the NONE action
                # Add bias away from action 0 (NONE)
                action_probs = np.ones(self.num_actions) * 0.8 / (self.num_actions - 1)
                action_probs[0] = 0.2  # Give NONE action lower probability
                action = np.random.choice(self.num_actions, p=action_probs)
            else:
                action = 0  # Only one action available
        else:
            # Greedy action selection
            action = int(np.argmax(self.Q[state]))
        
        self.last_state = state
        self.last_action = action
        return action

    def update(self):
        if self.last_state is None or self.last_action is None:
            return
        
        # Get current observation and corresponding state
        obs = self.game.observe(self.agent)
        state = self._obs_to_state(obs)
        
        # Get external reward from the environment
        reward_from_game = self.game.reward(self.agent)

        # If reward_from_game is None, it might indicate an issue with the training loop order
        # (e.g., update called before step, or game.reset() rewards not overridden by step).
        # Default to 0.0 to prevent TypeError, but this could mask underlying issues.
        current_reward = 0.0
        if reward_from_game is not None:
            current_reward = reward_from_game
        # else:
            # Consider adding a log or warning here if None rewards are frequent,
            # as it points to a potential logical flaw in the interaction loop.
            # print(f"Warning: Agent {self.agent} received None reward from game. Using 0.0 for update.")

        # Apply reward shaping if enabled
        shaped_reward = current_reward # Initialize with a guaranteed float
        if self.use_reward_shaping and self.last_obs is not None:
            # Extract state information for both current and previous observations
            prev_state_features = self._extract_state_features(self.last_obs)
            curr_state_features = self._extract_state_features(obs)
            
            # If we have valid state features, apply reward shaping
            if prev_state_features and curr_state_features:
                # Provide a small reward for moving toward food
                if self._is_closer_to_food(curr_state_features, prev_state_features):
                    shaped_reward += 0.05  # Small bonus for moving toward food
                # Slight penalty for taking the NONE action
                if self.last_action == 0:  # NONE action
                    shaped_reward -= 0.01  # Small penalty to encourage movement
        
        # Optimistic initialization
        if state not in self.Q:
            self.Q[state] = np.ones(self.num_actions) * self.optimistic_init
        
        # Calculate Q-learning update with shaped reward
        best_next = np.max(self.Q[state])
        td_target = shaped_reward + self.gamma * best_next
        td_error = td_target - self.Q[self.last_state][self.last_action]
        self.Q[self.last_state][self.last_action] += self.alpha * td_error
        
        # Apply epsilon decay if learning is enabled
        if self.learn:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Update state tracking
        self.last_state = state
        self.last_action = None
        self.last_obs = obs
    
    def _extract_state_features(self, obs):
        """Extract key features from observation for reward shaping"""
        if not isinstance(obs, np.ndarray):
            return None
        
        if len(obs.shape) == 3:  # Grid-based observation
            # Try to identify agent position and food positions (similar to _obs_to_state)
            agent_pos = None
            food_pos = None
            
            if obs.shape[0] >= 2:
                # Find agent position
                agent_channel = obs[0]
                if agent_channel.max() > 0:
                    agent_pos = np.unravel_index(agent_channel.argmax(), agent_channel.shape)
                
                # Find food positions
                food_channel = obs[1] if obs.shape[0] > 1 else None
                if food_channel is not None and food_channel.max() > 0:
                    food_positions = np.transpose(np.nonzero(food_channel))
                    if len(food_positions) > 0:
                        if agent_pos:
                            # Find nearest food
                            distances = [np.sqrt((f[0] - agent_pos[0])**2 + (f[1] - agent_pos[1])**2) for f in food_positions]
                            nearest_idx = np.argmin(distances)
                            food_pos = tuple(food_positions[nearest_idx])
                        else:
                            # Just take the first food
                            food_pos = tuple(food_positions[0])
            
            return {'agent_pos': agent_pos, 'food_pos': food_pos}
        return None
    
    def _is_closer_to_food(self, curr_features, prev_features):
        """Check if the agent moved closer to food"""
        if (curr_features and prev_features and 
            curr_features.get('agent_pos') and prev_features.get('agent_pos') and 
            curr_features.get('food_pos') and prev_features.get('food_pos')):
            
            prev_agent_pos = prev_features['agent_pos']
            curr_agent_pos = curr_features['agent_pos']
            food_pos = curr_features['food_pos']  # Use current food position
            
            # Calculate distances
            prev_dist = np.sqrt((prev_agent_pos[0] - food_pos[0])**2 + (prev_agent_pos[1] - food_pos[1])**2)
            curr_dist = np.sqrt((curr_agent_pos[0] - food_pos[0])**2 + (curr_agent_pos[1] - food_pos[1])**2)
            
            # Return True if we're closer now
            return curr_dist < prev_dist
        
        return False

    def policy(self):
        obs = self.game.observe(self.agent)
        state = self._obs_to_state(obs)
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
        policy = np.zeros(self.num_actions)
        best_action = int(np.argmax(self.Q[state]))
        policy[best_action] = 1.0
        return policy
