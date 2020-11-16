from abc import ABC, abstractmethod
import numpy as np

from misc.base_types import Controller, ForwardModel
from misc.rolloutbuffer import RolloutBuffer, Rollout


class ForwardModelWithDefaults(ForwardModel, ABC):
    supports_stochastic = False

    def reset(self, observation):
        return None

    def got_actual_observation_and_env_state(self, *, observation, env_state=None, model_state=None):
        return None

    def rollout_generator(self, start_states, start_observations, horizon, policy, mode=None):
        states = start_states
        obs = start_observations
        for h in range(horizon):
            # Todo: passing the state here is not 100% correct (since not env_state)
            actions = policy.get_action(obs, state=states, mode=mode)
            next_obs, next_states, r = self.predict(observations=obs, states=states, actions=actions)
            yield obs, next_obs, actions, states, r
            states = next_states
            obs = next_obs

    def rollout_field_names(self):
        return "observations", "next_observations", "actions", "rewards"

    def predict_n_steps(self, *, start_observations: np.ndarray, start_states: np.ndarray, policy: Controller,
                        horizon) -> (RolloutBuffer, np.ndarray):
        # default implementation falls back to predict
        if start_observations.ndim != 2:
            raise AttributeError("call predict_n_steps with a batch of states")
        if len(start_observations) != len(start_states):
            raise AttributeError("number of observations and states have to be the same")

        all_obs, all_next_obs, all_actions, all_states, all_rewards = zip(*self.rollout_generator(start_states,
                                                                                                  start_observations,
                                                                                                  horizon,
                                                                                                  policy))
        all_obs = np.asarray(all_obs).transpose((1, 0, 2))
        all_next_obs = np.asarray(all_next_obs).transpose((1, 0, 2))
        all_actions = np.asarray(all_actions).transpose((1, 0, 2))
        all_rewards = np.asarray(all_rewards).transpose((1, 0, 2))

        rollouts = [
            Rollout.from_dict(observations=obs, actions=acts, next_observations=next_obs, rewards=rewards)
            for obs, acts, next_obs, rewards in zip(all_obs, all_actions, all_next_obs, all_rewards)
        ]

        return RolloutBuffer(rollouts=rollouts), all_states[-1]


class StochasticModel(ForwardModelWithDefaults, ABC):
    supports_stochastic = True

    @abstractmethod
    def predict_stochastic(self, *, observations, states, actions):
        pass


class SequentialEnsembleModel(StochasticModel, ABC):
    def __init__(self, *, number_of_models, shuffle_models, **kwargs):
        super().__init__(**kwargs)
        self.number_of_models = number_of_models
        self.shuffle_models = shuffle_models
        self.models = []

    def _get_deterministic_predictions(self, *, observations, states, actions):
        return [model.predict(observations=observations, states=states, actions=actions)
                for model in self.models]
