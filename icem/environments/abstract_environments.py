from abc import ABC, abstractmethod

import gym.spaces as spaces
import numpy as np

from misc.base_types import Env


class EnvWithDefaults(Env, ABC):
    def __init__(self, *, name, **kwargs):
        super().__init__(name=name, **kwargs)

    # noinspection PyUnusedLocal
    def cost_fn(self, observation, action, next_obs):
        # compute for all samples, along coordinate axis
        dist = np.linalg.norm((observation - self.goal_state) * self.goal_mask, axis=-1)
        return dist

    def reward_fn(self, observation, action, next_obs):
        return -self.cost_fn(observation, action, next_obs)

    def from_full_state_to_transformed_state(self, full_state):
        return full_state

    def reset_with_mode(self, mode):
        return self.reset()

    def get_fps(self):
        if hasattr(self, "dt"):
            return int(np.round(1.0 / self.dt))
        elif hasattr(self, "metadata") and "video.frames_per_second" in self.metadata:
            return self.metadata["video.frames_per_second"]
        else:
            raise NotImplementedError("Environent does not have a generic way to get FPS. Overwrite get_fps()")

    @staticmethod
    def filter_buffers_by_cost(buffers, costs, filtered_fraction):
        if filtered_fraction == 1:
            print('Trajectories are not pre-filtered.')
            return [buffer.flat for buffer in buffers]
        else:
            num = int(len(costs) * filtered_fraction)
            print(f'Pre-filtering (keeping) {filtered_fraction * 100:.2f}% of all trajectories in the memory.')
            idxs = [np.array(c['costs']).argsort()[:num] for c in costs]

            # Loop over time steps. Every step has a buffer of simulated trajectories.
            return [buffer.flat[idx] for buffer, idx in zip(buffers, idxs)]


class DiscreteActionReshaper(EnvWithDefaults, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sampler = self.action_space.sample
        self.action_space.sample = self.new_sample

    def new_sample(self):
        if isinstance(self.action_space, spaces.Discrete):
            return np.array([self.sampler()])
        elif isinstance(self.action_space, spaces.Box):
            return self.sampler()
        else:
            raise NotImplementedError("Got some weird shit ... {}".format(type(self.action_space)))

    def step(self, action):
        if isinstance(action, np.ndarray):
            return super().step(int(action))
        elif isinstance(action, int):
            return super().step(action)
        else:
            raise NotImplementedError("Got some weird shit ... {}".format(type(action)))


class GoalSpaceEnvironmentInterface(ABC):

    # should extract the desired goal from the observation (vectorized)
    @abstractmethod
    def goal_from_observation(self, observations):
        pass

    # should extract the achieved goal (current state in goal space)
    @abstractmethod
    def achieved_goal_from_observation(self, observations):
        pass

    # should overwrite the desired goal in the observations to the given goals
    #  The following relationship has to hold: goal_from_observation(returned) -> goals
    @abstractmethod
    def overwrite_goal_inplace(self, observations, goals):
        pass

    # Checks if current step resulted in success
    @abstractmethod
    def is_success(self, observation, action, next_obs):
        pass


class MaskedGoalSpaceEnvironmentInterface(GoalSpaceEnvironmentInterface, ABC):
    def __init__(self, *, name, goal_idx, achieved_goal_idx, sparse: bool, threshold=0.1):
        self.goal_idx = goal_idx
        self.achieved_goal_idx = achieved_goal_idx
        self.sparse = sparse
        self.threshold = threshold
        assert self.threshold >= 0

    def goal_from_observation(self, observations):
        return np.take(observations, self.goal_idx, axis=-1)

    def achieved_goal_from_observation(self, observations):
        return np.take(observations, self.achieved_goal_idx, axis=-1)

    def overwrite_goal_inplace(self, observations, goals):
        observations[:, self.goal_idx] = goals
        return observations

    def cost_fn(self, observation, action, next_obs):

        dist = np.linalg.norm(self.goal_from_observation(observation) -
                              self.achieved_goal_from_observation(observation), axis=-1)
        if self.sparse:
            cost = np.asarray(dist > self.threshold, dtype=np.float32)
        else:
            cost = dist
        return cost

    def is_success(self, observation, action, next_obs):

        dist = np.linalg.norm(self.goal_from_observation(next_obs) -
                              self.achieved_goal_from_observation(next_obs), axis=-1)

        is_success = np.asarray(dist <= self.threshold, dtype=np.float32)

        return is_success

    def reward_fn(self, observation, action, next_obs):
        cost = self.cost_fn(observation, action, next_obs)

        return -cost


class GroundTruthSupportEnv(EnvWithDefaults, ABC):

    def __init__(self, *, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.init_kwargs = kwargs

    def store_init_arguments(self, all_parameters):
        # hacky way to store the parameters that are used to construct the object
        # (which we need to create a copy without a copy operation, namely by called the constructor again)
        forbidden_parameters = ['name', 'self', '__class__', 'kwargs']
        self.init_kwargs.update({k: v for k, v in all_parameters.items() if k not in forbidden_parameters})
        if 'kwargs' in all_parameters:
            self.init_kwargs.update(
                {k: v for k, v in all_parameters['kwargs'].items() if k not in forbidden_parameters})

    # noinspection PyPep8Naming
    @abstractmethod
    def set_GT_state(self, state):
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def get_GT_state(self):
        pass

    @abstractmethod
    def set_state_from_observation(self, observation):
        pass

    # noinspection PyMethodMayBeStatic
    def compute_state_difference(self, state1, state2):
        return np.max(state1 - state2)

    # simulates one step of the env by resetting to the given state first (not the observation, but the env-state)
    def simulate(self, state, action):
        self.set_GT_state(state)
        new_obs, r, *_ = self.step(action)
        new_state = self.get_GT_state()
        return new_obs, new_state, r
