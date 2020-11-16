# only fully abstract types should go here. Every thing with implementation goes one down in the hierarchy
from typing import Sequence

import numpy as np

from abc import ABC, abstractmethod
import gym

from misc.rolloutbuffer import RolloutBuffer


class Env(gym.Env, ABC):
    goal_state = None
    goal_mask = None
    supports_live_rendering = True
    enable_rendering_at_init = False

    def __init__(self, *, name, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    @abstractmethod
    def cost_fn(self, observation, action, next_obs):
        pass

    @abstractmethod
    def reward_fn(self, observation, action, next_obs):
        pass

    @abstractmethod
    def reset_with_mode(self, mode):
        pass

    @abstractmethod
    def get_fps(self):
        pass

    def prepare_for_recording(self):
        pass


class Controller(ABC):
    needs_training = False
    needs_data = False
    has_state = False
    required_settings = []

    # noinspection PyUnusedLocal
    def __init__(self, *, env: Env):
        self.env = env

    @abstractmethod
    def get_action(self, obs, state, mode="train"):
        """Performs an action.
        :param obs: observation from environment
        :param state: some internal state from the environment that might be used
        :param mode: "train" or "eval" or "expert"
        """
        pass


class ForwardModel(ABC):
    supports_stochastic = False

    # noinspection PyUnusedLocal
    def __init__(self, *, env: Env):
        self.env = env

    @abstractmethod
    def train(self, buffer: RolloutBuffer):
        pass

    @abstractmethod
    def reset(self, observation):
        pass

    @abstractmethod
    def got_actual_observation_and_env_state(self, *, observation, env_state=None, model_state=None):
        pass

    @abstractmethod
    def rollout_generator(self, start_states, start_observations, horizon, policy, mode=None):
        pass

    @abstractmethod
    def rollout_field_names(self):
        pass

    @abstractmethod
    def predict(self, *, observations, states, actions) -> tuple:
        """
        predict one step for all observations (using internal states) and actions
        :return (new_obs, new_states, new_reward)
        :param actions: action vector (or list of vectors)
        :param states: internal states/hidden variables of model (or list of them)
        :param observations: observation vector (or list of them)
        """
        pass

    @abstractmethod
    def predict_n_steps(self, *, start_observations: np.ndarray,
                        start_states: Sequence, policy: Controller, horizon) -> (RolloutBuffer, np.ndarray):
        pass

    # noinspection PyMethodMayBeStatic
    def get_state(self, observation):
        return None

    def set_state(self, state):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class Pretrainer(ABC):
    @abstractmethod
    def get_data(self, env):
        """ returns the data needed to train a module
        """
        pass
