from base.game import SimultaneousGame, AgentID, ActionDict
import gymnasium as gym
from lbforaging.foraging.environment import ForagingEnv, Player, Action

# Store a copy of the envirornment

class ForagingCopy():
    def __init__(self, env:ForagingEnv):

        # field
        self.field = env.unwrapped.field.copy()

        # players
        self.players = [Player() for _ in env.unwrapped.players]
        for i in range(len(self.players)):
            self.players[i].controller = env.unwrapped.players[i].controller
            self.players[i].position = env.unwrapped.players[i].position
            self.players[i].level = env.unwrapped.players[i].level
            self.players[i].field_size = env.unwrapped.players[i].field_size
            self.players[i].score = env.unwrapped.players[i].score
            self.players[i].reward = env.unwrapped.players[i].reward
            self.players[i].history = env.unwrapped.players[i].history
            self.players[i].current_step = env.unwrapped.players[i].current_step

        # current step
        self.current_step = env.unwrapped.current_step

        # game over
        self._game_over = env.unwrapped._game_over

        # observations
        self.obs = env.unwrapped.test_make_gym_obs()

    def reset(self, env:ForagingEnv):

        # field
        env.unwrapped.field = self.field.copy()

        # players
        for i in range(len(self.players)):
            env.unwrapped.players[i].controller = self.players[i].controller
            env.unwrapped.players[i].position = self.players[i].position
            env.unwrapped.players[i].level = self.players[i].level
            env.unwrapped.players[i].field_size = self.players[i].field_size
            env.unwrapped.players[i].score = self.players[i].score
            env.unwrapped.players[i].reward = self.players[i].reward
            env.unwrapped.players[i].current_step = self.players[i].current_step

        # current step
        env.unwrapped.current_step = self.current_step
        env.unwrapped._game_over = self._game_over
        env.unwrapped._gen_valid_moves()

        # observations and infos
        obs = [o.copy() for o in self.obs]
        infos = {}

        return obs, infos

class Foraging(SimultaneousGame):
    def __init__(self, config: str | None = None, seed: int | None = None):
    
        # environment
        if config is None:
            config = "Foraging-8x8-2p-1f-v3"    
        self.env = gym.make(config)
        self.env_copy = None

        # action set
        self.action_set = [a.name for a in list(Action)]

        # seed
        self.seed = seed

        # agents
        self.agents = ["agent_" + str(r) for r in range(len(self.env.unwrapped.players))]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents()))))

        self.observations = None
        self.rewards = None
        self.terminations = None
        self.truncations = None
        self.infos = None

        # actions
        self.action_spaces = {
            agent: self.env.action_space[i] for i, agent in enumerate(self.agents)
        }   
            
    # num_agents
    def num_agents(self):
        return len(self.agents)
    
    # step
    def step(self, actions: ActionDict) -> tuple[dict, dict, dict, dict, dict]:
        # actions
        action = []
        for agent in self.agents:
            action.append(actions[agent])
        action = tuple(action)

        # step
        obs, rewards, done, truncated, info = self.env.step(action=action)
        self.current_step = self.env.unwrapped.current_step

        # update observations, rewards, terminations, truncations, infos
        for i, agent in enumerate(self.agents):
            self.rewards[agent] = rewards[i]
            self.observations[agent] = {
                'observation': obs[i].copy(),   # observation of the agent
                'action': action                # joint action of all agents
            }
            self.terminations[agent] = done
            self.truncations[agent] = truncated
            self.infos[agent] = info
        
        self._done = done
        self._truncated = truncated
        
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    # reset
    def _reset(self, obs: tuple):
        self.observations = dict(map(lambda agent: (agent, {'observation': obs[self.agent_name_mapping[agent]], 'action': None}), self.agents))
        self.rewards = dict(map(lambda agent: (agent, 0), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))
        self._done = False
        self._truncated = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        if self.env_copy is None: 
            if seed is None:
                seed = self.seed
            # first time - reset the environment and store a copy
            obs, _ = self.env.reset(seed=seed, options=options)
            self.env_copy = ForagingCopy(self.env)
        else:
            # environment exists - restore the initial environment
            obs, _ = self.env_copy.reset(self.env)
        self.current_step = self.env.unwrapped.current_step
        # reset 
        self._reset(obs)
    
    # get observation
    def observe(self, agent: AgentID):
        # check if agent is valid
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not valid. Valid agents are: {self.agents}")
        # get observation
        observation = self.observations[agent]['observation']
        return observation
    
    # get actions
    def observe_action(self, agent: AgentID):
        # check if agent is valid
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not valid. Valid agents are: {self.agents}")
        # get action
        action = self.observations[agent]['action']
        return action
    
    # render
    def render(self):
        self.env.render()   

    # close
    def close(self):
        self.env.close()

    # done
    def done(self):
        return self._done