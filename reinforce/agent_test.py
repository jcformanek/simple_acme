import gym
import acme.wrappers as wrappers
import acme.specs as specs
from agent import RandomAgent
from acme.utils import loggers
import acme


def test_reinforce():
  # Create LunarLanderContinuous environment to test with.
  environment = gym.make('LunarLanderContinuous-v2')
  environment = wrappers.GymWrapper(environment)  # Convert to dm_env interface.
  environment = wrappers.SinglePrecisionWrapper(environment)
  spec = specs.make_environment_spec(environment)

  # loggers
  env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=1.)

  # Construct the agent.
  agent = REINFORCEAgent(environment_spec=spec)

  # Try running the environment loop. We have no assertions here because all
  # we care about is that the agent runs without raising any errors.
  loop = acme.EnvironmentLoop(environment, agent, logger=env_loop_logger)
  loop.run(num_episodes=200)


if __name__=="__main__":
  test_random()