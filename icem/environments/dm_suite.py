import collections

from dm_control.rl.control import flatten_observation
from dm_control.suite import cheetah, reacher
from dm_control.suite.utils import randomizers
from dm_env import _environment as environment

from .dm2gym import *
from helpers import sin_and_cos_to_radians


class CartPoleSuite(DmControlWrapper):
    domain_name = "cartpole"
    supports_live_rendering = False
    goal_state = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])  # slider_pos, cos, sin, slider_vel, thetadot
    goal_mask = np.array([[0.0, 1.0, 1.0, 0.0, 0.0]])  # we just care about the angle

    # noinspection PyProtectedMember
    def set_state_from_observation(self, observation):
        x_pos, cos_theta, sin_theta, *vels = observation
        theta = sin_and_cos_to_radians(sin_theta, cos_theta)
        # noinspection PyProtectedMember
        self.dmcenv._physics.set_state(np.array([x_pos, theta] + vels))
        self.dmcenv._physics.after_reset()


# noinspection PyProtectedMember
class ReacherSuite(DmControlWrapper):
    domain_name = "reacher"

    def cost_fn(self, states, actions, next_states):
        fingertip_target_diff = states[:, 2:4]
        return np.linalg.norm(fingertip_target_diff, axis=1)

    def set_state_from_observation(self, observation):
        physics_state = np.concatenate([observation[0:2], observation[4:6]], axis=-1)
        self.dmcenv._physics.set_state(physics_state)
        self.dmcenv._physics.after_reset()


def reacher_initialize_episode_with_goal(self, physics, mode=None):
    """Sets the state of the environment at the start of each episode."""
    physics.named.model.geom_size['target', 0] = self._target_size
    if mode == 'evaluate':
        randomizers.randomize_limited_and_rotational_joints(
            physics, FixedPosPlusUniformRandom(1, std=self.init_position_std_train))
    else:
        randomizers.randomize_limited_and_rotational_joints(
            physics, FixedPosPlusUniformRandom(1, std=self.init_position_std_eval))

    # Randomize target position
    if self.goal_xcoor is not None and self.goal_ycoor is not None:
        physics.named.model.geom_pos['target', 'x'] = self.goal_xcoor
        physics.named.model.geom_pos['target', 'y'] = self.goal_ycoor
    else:
        angle = self.random.uniform(0, 2 * np.pi)
        radius = self.random.uniform(.05, .20)
        physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
        physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)
    super(reacher.Reacher, self).initialize_episode(physics)


class FixedPosPlusUniformRandom(np.random.RandomState):
    def __init__(self, seed, std, random=None):
        super().__init__(seed)
        self.std = std
        self.rand_gen = random or np.random

    def rand(self, d0, d1, *more, **kwargs):
        return super().rand(d0, d1, *more, **kwargs) + self.rand_gen.uniform(-self.std, self.std)

    def uniform(self, low=0.0, high=1.0, size=None):
        return super().uniform(low=0.0, high=1.0, size=None) + self.rand_gen.uniform(-self.std, self.std)

    def randn(self, d0, d1, *more, **kwargs):
        return super().randn(d0, d1, *more, **kwargs) + self.rand_gen.uniform(-self.std, self.std)


class RestrictedReacherSuite(ReacherSuite):

    # def __new__(cls, **kwargs):
    #     reacher.Reacher.initialize_episode = reacher_initialize_episode_with_goal
    #     return ReacherSuite.__new__(cls)

    def __init__(self, *, name, task_name, task_kwargs=None, goal_xcoor=-0.15, goal_ycoor=-0.1,
                 init_position_std_train=0.05, init_position_std_eval=0.1,
                 visualize_reward=True, render_mode="human", **kwargs):
        reacher.Reacher.initialize_episode = reacher_initialize_episode_with_goal
        super().__init__(name=name, task_name=task_name, task_kwargs=task_kwargs, visualize_reward=visualize_reward,
                         render_mode=render_mode, **kwargs)
        # we have to save all arguments in order to create a "copy" of the env later
        # (from string and *kwargs) (e.g. for GTModel)
        self.store_init_arguments(locals())

        reacher.Reacher.goal_xcoor = goal_xcoor
        reacher.Reacher.goal_ycoor = goal_ycoor
        reacher.Reacher.init_position_std_train = init_position_std_train
        reacher.Reacher.init_position_std_eval = init_position_std_eval

    def reset_with_mode(self, mode):
        """Starts a new episode and returns the first `TimeStep`."""
        self.dmcenv._reset_next_step = False
        self.dmcenv._step_count = 0
        with self.dmcenv._physics.reset_context():
            self.dmcenv._task.initialize_episode(self.dmcenv._physics, mode=mode)

        observation = self.dmcenv._task.get_observation(self.dmcenv._physics)
        if self.dmcenv._flat_observation:
            observation = flatten_observation(observation)

        self.timestep = environment.TimeStep(
            step_type=environment.StepType.FIRST,
            reward=None,
            discount=None,
            observation=observation)

        return self._get_observation()


class DoubleIntSuite(DmControlWrapper):
    domain_name = "point_mass"
    supports_live_rendering = False

    goal_state = np.array([[0.0, 0.0, 0.0, 0.0]])
    goal_mask = np.array([[1.0, 1.0, 0.0, 0.0]])

    def __init__(self, *, name, task_name, task_kwargs=None,
                 visualize_reward=True, render_mode="human", init_std=None, **kwargs):
        super().__init__(name=name, task_name=task_name, task_kwargs=task_kwargs, visualize_reward=visualize_reward,
                         render_mode=render_mode, **kwargs)
        self.store_init_arguments(locals())
        self.init_std = init_std

    def reset_with_mode(self, mode):
        obs = super().reset()
        # if self.init_std is not None:
        #     obs[0] = 0.2 + np.random.randn() * self.init_std
        #     obs[1] = 0.1 + np.random.randn() * self.init_std
        #     self.set_GT_state(np.insert(obs, 0, 0))
        return obs

    # noinspection PyProtectedMember
    def set_state_from_observation(self, observation):
        self.dmcenv.reset()
        self.dmcenv._physics.set_state(np.array(observation))
        self.dmcenv._physics.after_reset()

    def from_full_state_to_transformed_state(self, full_state):
        position = full_state[:, :2]
        velocity = full_state[:, 2:4]
        next_position = position + self.dmcenv._physics.timestep() * self.dmcenv._n_sub_steps * velocity
        transformed_state = np.concatenate((position, next_position), axis=1)
        return transformed_state

    @staticmethod
    def filter_buffers_by_cost(buffers, costs, filtered_fraction=1):
        if filtered_fraction == 1:
            filtered_fraction = 50
        return [traj for costs, trajectories in zip(costs, buffers) for n, traj
                in enumerate(trajectories) if
                costs['costs'][n] < (1 + filtered_fraction) * (costs['best_cost'] + 1e-6)]


class RestrictedDoubleIntSuite(DoubleIntSuite):
    def __init__(self, *, name, task_name, task_kwargs=None,
                 visualize_reward=True, render_mode="human", init_std=None, init_std_eval=None, **kwargs):
        super().__init__(name=name, task_name=task_name, task_kwargs=task_kwargs, visualize_reward=visualize_reward,
                         render_mode=render_mode, **kwargs)
        self.store_init_arguments(locals())
        self.init_std = init_std
        self.init_std_eval = init_std_eval
        self.rand_generator_train = np.random.RandomState(1)
        self.rand_generator_eval = np.random.RandomState(2)

    def reset_with_mode(self, mode):
        obs = super().reset()
        if self.init_std is not None:
            obs[0] = 0.2 + self.rand_generator_train.uniform(-self.init_std, self.init_std)
            obs[1] = 0.1 + self.rand_generator_train.uniform(-self.init_std, self.init_std)
            self.set_GT_state(np.insert(obs, 0, 0))
        if self.init_std_eval is not None and mode == "evaluate":
            obs[0] = 0.2 + self.rand_generator_eval.uniform(-self.init_std_eval, self.init_std_eval)
            obs[1] = 0.1 + self.rand_generator_eval.uniform(-self.init_std_eval, self.init_std_eval)
            self.set_GT_state(np.insert(obs, 0, 0))
        return obs


# This is a Monkey patch function for the cheetah within the DM suite. Don't wonder about the "self"
# noinspection PyUnusedLocal
def get_observation_with_position(self, physics):
    """Returns an observation of the state, INCLUDING horizontal position."""
    obs = collections.OrderedDict()
    obs["position"] = physics.data.qpos.copy()
    obs["velocity"] = physics.velocity()
    return obs


class HalfCheetahSuite(DmControlWrapper):
    domain_name = "cheetah"
    supports_live_rendering = False

    cheetah.Cheetah.get_observation = get_observation_with_position

    def __init__(self, *, name, task_name, task_kwargs=None,
                 visualize_reward=True, render_mode="human", penalise_flipping, **kwargs):
        super().__init__(name=name, task_name=task_name, task_kwargs=task_kwargs, visualize_reward=visualize_reward,
                         render_mode=render_mode, **kwargs)
        self.store_init_arguments(locals())
        self.penalise_flipping = penalise_flipping

    def cost_fn(self, states, actions, next_states=None):
        if len(states.shape) > 1:
            is_single_state = False
        else:
            is_single_state = True

        if is_single_state:
            states = states[None, ...]
            actions = actions[None, ...]

        if states.shape[-1] == 18:
            root_angle = states[..., 2]
            velocity = states[..., 9]
        elif states.shape[-1] == 17:
            root_angle = states[..., 1]
            velocity = states[..., 8]
        else:
            raise ValueError(f'Got state of dimension {states.shape[-1]}. Possible dimensions are 17 or 18.')

        scores = np.zeros(list(actions.shape)[:-1])

        if self.penalise_flipping:
            heading_penalty_factor = 10
            scores += (root_angle > math.pi/2) * heading_penalty_factor
            scores += (root_angle < -math.pi / 2) * heading_penalty_factor

        scores += 0.1 * (np.sum(actions ** 2, axis=-1))
        # scores -= (next_obs[:, 0] - states[:, 0]) / self.dt
        scores -= velocity

        if is_single_state:
            scores = scores[0]

        return scores  # [batch_size, models]

    def set_state_from_observation(self, observation):
        self.dmcenv.reset()
        self.dmcenv._physics.set_state(np.array(observation))
        self.dmcenv._physics.after_reset()


class SwimmerSuite(DmControlWrapper):
    domain_name = "swimmer"
    supports_live_rendering = False

    def cost_fn(self, states, actions, next_states):
        nose_target_diff = states[:, -20:-18]
        return np.linalg.norm(nose_target_diff, axis=1)

    def set_state_from_observation(self, observation):
        raise NotImplementedError
