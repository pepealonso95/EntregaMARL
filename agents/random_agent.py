import numpy as np
from base.game import SimultaneousGame, AgentID
from base.agent import Agent

class RandomAgent(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        action_space = game.action_spaces[agent]
        num_actions = action_space.n  # Use Gymnasium action space
        if initial is None:
            self._policy = np.full(num_actions, 1/num_actions)
        else:
            self._policy = initial

    def action(self):
        # For Gymnasium environments, sample directly from action space
        return int(np.random.choice(len(self._policy), p=self._policy))  # Ensure we return an integer
    
    def policy(self):
        return self._policy
