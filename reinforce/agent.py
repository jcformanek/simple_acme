import tensorflow as tf

from acme.agents import agent
from acme import specs

from learning import REINFORCELearner
import reverb

class REINFORCEAgent(agent.Agent):
    """
    REINFORCE agent.
    """

    def __init__(self, 
        environment_spec: specs.EnvironmentSpec,
        policy_network: snt.Module,
        discount: float = 0.99,
        policy_optimizer: snt.Optimizer = None,
        transitions_per_update: int = 4000,
    ):

        self._policy_network = policy_network

        # Get observation and action specs.
        action_spec = environment_spec.actions

        buffer_table = reverb.Table.queue(name='Buffer', max_size=timesteps_per_update)
        buffer_server = reverb.Server(tables=[buffer_table], port=8000)
        buffer_client = reverb.Client('localhost:8000')

        actor = #Feed forward actor (policy_network, buffer)
        learner = REINFORCELearner(policy_network, buffer_client, timesteps_per_update, discount)
)

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=1,
            observations_per_step=1)