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
from acme.tf import networks

class DDPGAgent(agent.Agent):
    """DDPG agent."""

    def __init__(self,
        environment_spec: specs.EnvironmentSpec,
        policy_network: snt.Module,
        critic_network: snt.Module,
        target_update_period: int = 100,
        discount: float = 0.99,
        max_replay_size: float = int(1e6),
        batch_size: int = 256,
    ):
        target_policy_network = copy.deepcopy(policy_network)
        target_critic_network = copy.deepcopy(critic_network)

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

        # Ensure that we create the variables before proceeding (maybe not needed).
        tf2_utils.create_variables(policy_network, [environment_spec.observations])
        tf2_utils.create_variables(critic_network, [environment_spec.observations, environment_spec.actions])
        tf2_utils.create_variables(target_policy_network, [environment_spec.observations])
        tf2_utils.create_variables(target_critic_network, [environment_spec.observations, environment_spec.actions])

        # Create the actor which defines how we take actions.
        # Create the behavior policy.
        behavior_network = snt.Sequential([
            policy_network,
            networks.ClippedGaussian(0.3), # sigma=0.3
            networks.TanhToSpec(environment_spec.actions)
        ])
        actor = actors.FeedForwardActor(behavior_network, adder)

        learner = learning.DDPGLearner(
                                    policy_network=policy_network,
                                    critic_network=critic_network,
                                    target_policy_network=target_policy_network,
                                    target_critic_network=target_critic_network,
                                    replay_client=replay_client,
                                    dataset=dataset,
                                    discount=discount,
                                    target_update_period=target_update_period)

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=batch_size,
            observations_per_step=1)

    def update(self):
        super().update()