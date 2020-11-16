from abc import ABCMeta, abstractmethod
from typing import Sequence

from misc.base_types import Controller
from environments.abstract_environments import GroundTruthSupportEnv
from .abstract_models import ForwardModelWithDefaults
from environments import env_from_string
import numpy as np
from misc.rolloutbuffer import RolloutBuffer, Rollout
from misc.seeding import Seeding


class AbstractGroundTruthModel(ForwardModelWithDefaults, metaclass=ABCMeta):

    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def get_state(self, observation):
        pass


class GroundTruthModel(AbstractGroundTruthModel):
    simulated_env: GroundTruthSupportEnv

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.env, GroundTruthSupportEnv):
            self.simulated_env = env_from_string(self.env.name, **self.env.init_kwargs)
            self.simulated_env.reset()
            self.is_trained = True
        else:
            raise NotImplementedError("Environment does not support ground truth forward model")

    def close(self):
        self.simulated_env.close()

    def train(self, buffer):
        pass

    def reset(self, observation):
        self.simulated_env.set_state_from_observation(observation)
        return self.get_state(observation)

    def got_actual_observation_and_env_state(self, *, observation, env_state=None, model_state=None):
        if env_state is None:
            self.simulated_env.set_state_from_observation(observation)
            return self.simulated_env.get_GT_state()
        else:
            return env_state

    def set_state(self, state):
        self.simulated_env.set_GT_state(state)

    def get_state(self, observation):
        return self.simulated_env.get_GT_state()

    def predict(self, *, observations, states, actions):
        def state_to_use(observation, state):
            if state is None:
                # This is an inefficiency as we set the state twice (typically not using state=None for GT models)
                self.simulated_env.set_state_from_observation(observation)
                return self.simulated_env.get_GT_state()
            else:
                return state

        if observations.ndim == 1:
            return self.simulated_env.simulate(state_to_use(observations, states), actions)
        elif states is None:
            states = [None] * len(observations)
        next_obs, next_states, rs = zip(*[self.simulated_env.simulate(state_to_use(obs, state), action)
                                          for obs, state, action in zip(observations, states, actions)])
        return np.asarray(next_obs), next_states, np.asarray(rs)

    def predict_n_steps(self, *, start_observations: np.ndarray, start_states: Sequence,
                        policy: Controller, horizon) -> (RolloutBuffer, Sequence):
        # here we want to step through the envs in the direction of time
        if start_observations.ndim != 2:
            raise AttributeError(f"call predict_n_steps with a batches (shape: {start_observations.shape})")
        if len(start_observations) != len(start_states):
            raise AttributeError("number of observations and states have to be the same")

        def perform_rollout(start_obs, start_state):
            self.simulated_env.set_GT_state(start_state)
            obs = start_obs
            for h in range(horizon):
                action = policy.get_action(obs, None)
                next_obs, r, _, _ = self.simulated_env.step(action)
                yield obs, next_obs, action, r
                obs = next_obs

        fields = self.rollout_field_names()

        def rollouts_generator():
            for obs_state in zip(start_observations, start_states):
                trans = perform_rollout(*obs_state)
                yield Rollout(field_names=fields, transitions=trans), self.simulated_env.get_GT_state()

        rollouts, states = zip(*rollouts_generator())

        return RolloutBuffer(rollouts=rollouts), states

    def save(self, path):
        pass

    def load(self, path):
        pass
