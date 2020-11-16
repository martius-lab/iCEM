from controllers.abstract_controller import Controller


class RndController(Controller):
    def __init__(self, *, action_sampler_params, **kwargs):
        super().__init__(**kwargs)
        self.action_sampler_params = action_sampler_params
        self.previous_action = self.env.action_space.sample()
        self.counter = 0

    def sample(self):
        """ Randomly samples an action uniformly from the action space,
        and keeps it unchanged for a number of steps equals to self.action_change_freq"""
        if self.counter < self.action_sampler_params.action_change_frequency:
            self.counter += 1
            return self.previous_action
        else:
            self.counter = 0
            self.previous_action = self.env.action_space.sample()
            return self.env.action_space.sample()

    def get_action(self, obs, state, mode="train"):
        return self.sample()
