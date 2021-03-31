import acme
from acme.utils import loggers
import tensorflow as tf
import reverbe

class REINFORCELearner(acme.Learner):

    def __init__(self,
        policy_network: snt.Module,
        buffer_client: reverbe.Client,
        timesteps_per_update: int,
        discount: float = 0.99,
        policy_optimizer: snt.Optimizer = None,
        logger: loggers.Logger = None
    ):
        self._policy_network = policy_network

        self._discount = discount

        self._timesteps_per_update = timesteps_per_update

        self._buffer_client = buffer_client

        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)

        self._variables = {'policy': policy_network.variables}

        self._logger = loggers.make_default_logger('learner')
        

    def step(self):
        """Run learning step."""

        sample = buffer_client.sample('Buffer', num_samples=self._timesteps_per_update)
        obs, act, rew, done, ret = sample.data

        with tf.GradientTape() as tape:
            pi, logp = self._policy_network(obs, act)
            loss_pi = -tf.reduce_mean(logp * ret)

        policy_variables = self._policy_network.trainable_variables
        policy_gradients = tape.gradient(policy_loss, policy_variables)
        self._policy_optimizer.apply(policy_gradients, policy_variables)


    def get_variables(self, names):
        """Return policy network variables."""
        return self._policy_network.trainable_variables
    