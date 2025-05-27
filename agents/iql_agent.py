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
        self.epsilon = getattr(config, 'epsilon', 0.9) if config is not None else 0.9
        self.min_epsilon = getattr(config, 'min_epsilon', 0.01) if config is not None else 0.01
        self.epsilon_decay = getattr(config, 'epsilon_decay', 0.9999) if config is not None else 0.9999
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
        self.last_obs = None  # Store previous observation for reward shaping
        self.debug_counter = 0

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_obs = None
        self.debug_counter = 0

    def _obs_to_state(self, obs):
        """
        Enhanced state representation for the Foraging environment.
        """
        if not isinstance(obs, np.ndarray):
            return tuple([hash(str(obs))])
        
        if obs.ndim == 3:  # Grid observation (height, width, channels)
            # Extract agent position (usually in the center)
            agent_pos = (obs.shape[0] // 2, obs.shape[1] // 2)
            
            # Find food locations and other agents
            food_channel = 1  # Assuming channel 1 has food information
            agent_channel = 0  # Assuming channel 0 has agent information
            
            state_features = []
            
            if obs.shape[2] > food_channel:
                food_layer = obs[:, :, food_channel]
                food_positions = np.argwhere(food_layer > 0)
                
                if len(food_positions) > 0:
                    # Find closest food
                    distances = [np.sqrt(((pos[0] - agent_pos[0])**2) + 
                                        ((pos[1] - agent_pos[1])**2)) 
                                for pos in food_positions]
                    closest_idx = np.argmin(distances)
                    closest_food = tuple(food_positions[closest_idx])
                    
                    # Direction vector to closest food (discretized to 8 directions)
                    dx = closest_food[1] - agent_pos[1]
                    dy = closest_food[0] - agent_pos[0]
                    direction = self._discretize_direction(dx, dy)
                    distance = min(int(distances[closest_idx]), 7)  # Cap distance
                    
                    state_features.extend([direction, distance])
                    
                    # Count nearby food items (within 2 steps)
                    nearby_food = sum(1 for d in distances if d <= 2)
                    state_features.append(min(nearby_food, 3))
                else:
                    state_features.extend([-1, -1, 0])  # No food visible
            else:
                state_features.extend([-1, -1, 0])
            
            # Add information about other agents if available
            if obs.shape[2] > agent_channel:
                agent_layer = obs[:, :, agent_channel]
                other_agent_positions = np.argwhere(agent_layer > 0)
                # Filter out self position
                other_agent_positions = [pos for pos in other_agent_positions 
                                       if not (pos[0] == agent_pos[0] and pos[1] == agent_pos[1])]
                
                if len(other_agent_positions) > 0:
                    # Direction to closest other agent
                    distances = [np.sqrt(((pos[0] - agent_pos[0])**2) + 
                                        ((pos[1] - agent_pos[1])**2)) 
                                for pos in other_agent_positions]
                    closest_idx = np.argmin(distances)
                    closest_agent = tuple(other_agent_positions[closest_idx])
                    
                    dx = closest_agent[1] - agent_pos[1]
                    dy = closest_agent[0] - agent_pos[0]
                    agent_direction = self._discretize_direction(dx, dy)
                    agent_distance = min(int(distances[closest_idx]), 7)
                    
                    state_features.extend([agent_direction, agent_distance])
                else:
                    state_features.extend([-1, -1])  # No other agents visible
            else:
                state_features.extend([-1, -1])
                
            return tuple(state_features)
        
        # Fallback for other observation types
        return tuple(obs.flatten()[:10])  # Take first 10 elements

    def _discretize_direction(self, dx, dy):
        """Convert dx, dy to a discrete direction (0-7)."""
        angle = np.arctan2(dy, dx)
        # Convert to 0-7 range (8 directions)
        direction = int(((angle + np.pi) / (2 * np.pi) * 8) % 8)
        return direction

    def action(self):
        obs = self.game.observe(self.agent)
        self.last_obs = obs

        state = self._obs_to_state(obs)
        
        if state not in self.Q:
            self.Q[state] = np.ones(self.num_actions) * self.optimistic_init
        
        if self.learn and np.random.rand() < self.epsilon:
            # Uniform random exploration
            action = np.random.randint(0, self.num_actions)
        else:
            # Greedy action selection
            action = int(np.argmax(self.Q[state]))
        
        self.last_state = state
        self.last_action = action
        return action

    def update(self):
        if self.last_state is None or self.last_action is None or self.last_obs is None:
            return
        
        current_obs = self.game.observe(self.agent)
        reward_from_game = self.game.reward(self.agent)

        current_reward = 0.0
        if reward_from_game is not None:
            current_reward = float(reward_from_game)

        shaped_reward = current_reward
        if self.use_reward_shaping:
            prev_state_features = self._extract_state_features(self.last_obs)
            curr_state_features = self._extract_state_features(current_obs)
            
            if prev_state_features and curr_state_features and \
               prev_state_features.get('food_pos') and curr_state_features.get('food_pos'):
                
                if self._is_closer_to_food(curr_state_features, prev_state_features):
                    shaped_reward += 0.05  # Increased reward for moving toward food
                elif current_reward == 0:  # Only penalize if no actual reward was received
                    shaped_reward -= 0.01  # Small penalty for moving away from food
        
        current_q_state = self._obs_to_state(current_obs)
        
        if current_q_state not in self.Q:
            self.Q[current_q_state] = np.ones(self.num_actions) * self.optimistic_init
        
        best_next_q = np.max(self.Q[current_q_state])
        td_target = shaped_reward + self.gamma * best_next_q
        td_error = td_target - self.Q[self.last_state][self.last_action]
        self.Q[self.last_state][self.last_action] += self.alpha * td_error

    def _extract_state_features(self, obs):
        """Extract key features (agent_pos relative to its view, food_pos relative to its view)
           from a partially observable observation for reward shaping.
           Assumes obs is the agent's local view, e.g., (height, width, channels)."""
        if not isinstance(obs, np.ndarray) or obs.ndim != 3:
            return None

        agent_relative_pos = (obs.shape[0] // 2, obs.shape[1] // 2)

        food_channel_idx = 1 
        
        if obs.shape[2] <= food_channel_idx:
            return {'agent_pos': agent_relative_pos, 'food_pos': None}

        food_layer = obs[:, :, food_channel_idx]
        food_locations_relative = np.argwhere(food_layer > 0) 

        if len(food_locations_relative) == 0:
            return {'agent_pos': agent_relative_pos, 'food_pos': None}

        distances_to_food = np.linalg.norm(food_locations_relative - np.array(agent_relative_pos), axis=1)
        nearest_food_idx = np.argmin(distances_to_food)
        nearest_food_relative_pos = tuple(food_locations_relative[nearest_food_idx])
        
        return {'agent_pos': agent_relative_pos, 'food_pos': nearest_food_relative_pos}

    def _is_closer_to_food(self, curr_features, prev_features):
        """Check if the agent moved closer to the (nearest) food within its view."""
        if not (curr_features and prev_features and
                curr_features.get('agent_pos') and prev_features.get('agent_pos') and
                curr_features.get('food_pos') and prev_features.get('food_pos')):
            return False

        prev_agent_center_in_view = np.array(prev_features['agent_pos'])
        prev_food_loc_in_view = np.array(prev_features['food_pos'])
        
        curr_agent_center_in_view = np.array(curr_features['agent_pos'])
        curr_food_loc_in_view = np.array(curr_features['food_pos'])

        prev_dist_to_food = np.linalg.norm(prev_food_loc_in_view - prev_agent_center_in_view)
        curr_dist_to_food = np.linalg.norm(curr_food_loc_in_view - curr_agent_center_in_view)

        return curr_dist_to_food < prev_dist_to_food

    def policy(self):
        obs = self.game.observe(self.agent)
        state = self._obs_to_state(obs)
        if state not in self.Q:
            return np.ones(self.num_actions) / self.num_actions

        policy_probs = np.zeros(self.num_actions)
        best_action = int(np.argmax(self.Q[state]))
        policy_probs[best_action] = 1.0
        return policy_probs
