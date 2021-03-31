import gym
import acme.wrappers as wrappers
import acme.specs as specs
from agent import DDPGAgent
from acme.utils import loggers
import acme
import sonnet as snt
from acme.agents.tf import actors
from acme.tf import networks as networks



def test_ddpg():
    # Create LunarLanderContinuous environment to test with.
    environment = gym.make('MountainCarContinuous-v0')
    environment = wrappers.GymWrapper(environment)  # Convert to dm_env interface.
    environment = wrappers.SinglePrecisionWrapper(environment)
    spec = specs.make_environment_spec(environment)

    # Network
    policy_net = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([50, 50]),
        networks.NearZeroInitializedLinear(spec.actions.shape[0]),
        networks.TanhToSpec(spec.actions)])

    critic_net = networks.CriticMultiplexer(
      critic_network=snt.nets.MLP([50, 50, 1]))

    # Construct the agent.
    agent = DDPGAgent(
                environment_spec=spec,
                policy_network=policy_net,
                critic_network=critic_net)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=1000)


if __name__=="__main__":
    test_ddpg()