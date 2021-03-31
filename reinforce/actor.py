import snt

class REINFORCEActor(core.Actor):

    def __init__(self, 
                policy_network: snt.Module,
    ):
        self._policy_network = policy_network
        self.shape = action_spec.shape
        self.scale = action_spec.maximum - action_spec.minimum
        self.offset = action_spec.minimum
        self.dtype = action_spec.dtype

        self._trajectory_timesteps = []
        self._trajectory_actions = []

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        """Samples a random action."""
        action = self._policy_network(observation).squeeze().numpy()
        action = action.astype(self.dtype)
        return action

    def observe_first(self, timestep: dm_env.TimeStep):
        """
        """
        pass

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
    ):
        """
        If end of trajectory compute returns and then write 
        all timesteps to reverbe.
        If not end of trajectory just store timestep locally.
        """
        self._trajectory_timesteps.append(next_timestep)
        self._trajectory_actions.append(action)

        if next_timestep.done == True:

            rtgs = _compute_rewards_to_go(self._trajectory_timesteps)

        

    def update(self, wait: bool = False):
        """Do nothing. Don't need update any parameters 
        for random actors.
        """
        pass