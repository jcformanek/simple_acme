from acme.agents import agent
from acme import specs
import sonnet as snt
from acme.adders import reverb as adders
import trfl
from acme.tf import utils as tf2_utils
import learning
from acme.utils import loggers
import reverb
from acme import datasets
from acme.agents.tf import actors
import tensorflow as tf
import numpy as np
import copy

class DQNAgent(agent.Agent):
    """DQN agent."""

    def __init__(self,
        environment_spec: specs.EnvironmentSpec,
        network: snt.Module,
        max_replay_size: float = int(1e6),
        target_update_period: int = 100,
        discount: float = 0.99,
        batch_size: int = 256,
        epsilon_decrement: float = 1e-3,
        logger: loggers.Logger = None
    ):

        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(environment_spec))

        self._server = reverb.Server([replay_table], port=None)

        address = f'localhost:{self._server.port}'
        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address),
            n_step=1,
            discount=discount)

        # The dataset provides an interface to sample from replay.
        replay_client = reverb.TFClient(address)
        dataset = datasets.make_reverb_dataset(
            server_address=address,
            batch_size=batch_size,
            prefetch_size=4)

        # Create epsilon greedy policy network by default.
        self._epsilon = tf.Variable(1.0, trainable=False)
        self._epsilon_decrement = epsilon_decrement

        policy_network = snt.Sequential([
            network,
            lambda q: trfl.epsilon_greedy(q, epsilon=self._epsilon).sample()
        ])

        # Create a target network.
        target_network = copy.deepcopy(network)

        # Ensure that we create the variables before proceeding (maybe not needed).
        tf2_utils.create_variables(network, [environment_spec.observations])
        tf2_utils.create_variables(target_network, [environment_spec.observations])

        # Create the actor which defines how we take actions.
        actor = actors.FeedForwardActor(policy_network, adder)

        learner = learning.DQNLearner(
                                    network=network,
                                    target_network=target_network,
                                    replay_client=replay_client,
                                    dataset=dataset,
                                    discount=discount,
                                    target_update_period=target_update_period)

        self._logger = logger

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=batch_size,
            observations_per_step=1)

    def update(self):
        self._epsilon.assign_sub(self._epsilon_decrement)
        if self._epsilon.numpy() < 0.05:
            self._epsilon.assign(0.05)
        super().update()