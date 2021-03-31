import gym
import acme.wrappers as wrappers
import acme.specs as specs
from agent import DQNAgent
from acme.utils import loggers
import acme
import sonnet as snt
from acme.agents.tf import actors


def test_dqn():
    # Create LunarLanderContinuous environment to test with.
    environment = gym.make('CartPole-v1')
    environment = wrappers.GymWrapper(environment)  # Convert to dm_env interface.
    environment = wrappers.SinglePrecisionWrapper(environment)
    spec = specs.make_environment_spec(environment)

    # Network
    q_net = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([50, 50, spec.actions.num_values])])

    logger = loggers.TerminalLogger(label='env_loop', time_delta=2.)

    # Construct the agent.
    agent = DQNAgent(
                environment_spec=spec,
                network=q_net,
                logger=logger)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, logger=logger)
    loop.run(num_episodes=1000)


if __name__=="__main__":
    test_dqn()