import reverb
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
import acme
from acme.utils import loggers
import trfl
from acme.tf import losses

class DDPGLearner(acme.Learner):
    """DDPG learner."""

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network: snt.Module,
        target_policy_network: snt.Module,
        target_critic_network: snt.Module,
        discount: float,
        dataset: tf.data.Dataset,
        target_update_period = 100,
        replay_client: reverb.TFClient = None,
        learning_rate: float = 1e-3,
        logger: loggers.Logger = None
    ):

        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network

        self._policy_optimizer = snt.optimizers.Adam(learning_rate)
        self._critic_optimizer = snt.optimizers.Adam(learning_rate)

        self._discount = discount

        self._replay_client = replay_client
        self._iterator = iter(dataset)

        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)


    def step(self):

        inputs = next(self._iterator)
        o_tm1, a_tm1, r_t, d_t, o_t = inputs.data

        with tf.GradientTape() as tape:
            # Critic learning.
            q_tm1 = self._critic_network(o_tm1, a_tm1)
            q_t = self._target_critic_network(o_t, self._target_policy_network(o_t))

            # Squeeze into the shape expected by the td_learning implementation.
            q_tm1 = tf.squeeze(q_tm1)
            q_t = tf.squeeze(q_t)

            # Critic loss.
            critic_loss = trfl.td_learning(q_tm1, r_t, self._discount * d_t, q_t).loss
            critic_loss = tf.reduce_mean(critic_loss)

        critic_variables = self._critic_network.trainable_variables
        critic_gradients = tape.gradient(critic_loss, critic_variables)
        self._critic_optimizer.apply(critic_gradients, critic_variables)        

        with tf.GradientTape(persistent=True) as tape:

            dpg_a_t = self._policy_network(o_t)
            dpg_q_t = self._critic_network(o_t, dpg_a_t)

            policy_loss = losses.dpg(
                            dpg_q_t,
                            dpg_a_t,
                            tape=tape)

            policy_loss = tf.reduce_mean(policy_loss)

        policy_variables = self._policy_network.trainable_variables
        policy_gradients = tape.gradient(policy_loss, policy_variables)
        self._policy_optimizer.apply(policy_gradients, policy_variables)
        del tape

        # Target update
        online_variables = (
        *self._critic_network.variables,
        *self._policy_network.variables,
        )

        target_variables = (
            *self._target_critic_network.variables,
            *self._target_policy_network.variables,
        )

        # Make online -> target network update ops.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)
        self._num_steps.assign_add(1)

    def get_variables(self, names):
        return None