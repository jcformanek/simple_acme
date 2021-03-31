import tensorflow as tf

from acme.agents import agent
from acme import specs

from actor import RandomActor
from learning import RandomLearner

class RandomAgent(agent.Agent):
    """
    Random agent. Samples a random action from the environment 
    action-space at every timestep.
    """

    def __init__(self, 
        environment_spec: specs.EnvironmentSpec,
    ):

        # Get observation and action specs.
        action_spec = environment_spec.actions

        actor = RandomActor(action_spec)
        learner = RandomLearner()

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=1,
            observations_per_step=1)
