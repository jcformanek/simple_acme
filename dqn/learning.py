import reverb
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
import acme
from acme.utils import loggers
import trfl

class DQNLearner(acme.Learner):
    """DQN learner."""

    def __init__(
        self,
        network: snt.Module,
        target_network: snt.Module,
        dataset: tf.data.Dataset,
        discount: float,
        target_update_period: int,
        replay_client: reverb.TFClient = None,
        learning_rate: float = 1e-3,
        logger: loggers.Logger = None
    ):

        self._network = network
        self._optimizer = snt.optimizers.Adam(learning_rate)
        self._target_network = target_network
        self._target_update_period = target_update_period
        self._replay_client = replay_client
        self._discount = discount
        self._iterator = iter(dataset)
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)


    def step(self):
        inputs = next(self._iterator)
        o_tm1, a_tm1, r_t, d_t, o_t = inputs.data

        with tf.GradientTape() as tape:
            # Evaluate our networks.
            q_tm1 = self._network(o_tm1)
            q_t_value = self._target_network(o_t)
            
            pcont_t = tf.cast(d_t, q_tm1.dtype) * tf.cast(self._discount, q_tm1.dtype)

            loss, _ = trfl.qlearning(q_tm1, a_tm1, r_t, d_t, q_t_value)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self._network.trainable_variables)
        self._optimizer.apply(gradients, self._network.trainable_variables)

        # Periodically update the target network.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(self._network.variables, self._target_network.variables):
                dest.assign(src)

        self._num_steps.assign_add(1)

    def get_variables(self, names):
        return self._network.trainable_variables
