from acme import core
from acme import specs
from acme import types
from acme.testing import fakes
import dm_env
import numpy as np

class RandomActor(core.Actor):

    def __init__(self, 
                action_spec: specs.BoundedArray,
    ):
        self.shape = action_spec.shape
        self.scale = action_spec.maximum - action_spec.minimum
        self.offset = action_spec.minimum
        self.dtype = action_spec.dtype

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        """Samples a random action."""
        action = np.random.random(self.shape) * self.scale + self.offset
        action = action.astype(self.dtype)
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        """Do nothing. Don't need to store transitions 
        for random actors.
        """
        pass

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
    ):
        """Do nothing. Don't need to store transitions 
        for random actors.
        """
        pass

    def update(self, wait: bool = False):
        """Do nothing. Don't need update any parameters 
        for random actors.
        """
        pass