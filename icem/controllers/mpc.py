import os
import pickle
from abc import ABC, abstractmethod
from typing import Union
from warnings import warn

import allogger
import colorednoise
import numpy as np
from gym import spaces
from scipy.stats import truncnorm

from controllers.abstract_controller import ModelBasedController, StatefulController, OpenLoopPolicy
from environments.abstract_environments import GroundTruthSupportEnv
from models import AbstractGroundTruthModel
from misc.rolloutbuffer import RolloutBuffer
from controllers.random import RndController


# abstract MPC controller
class MpcController(ModelBasedController, StatefulController, ABC):
    def __init__(self, *, horizon, num_simulated_trajectories, factor_decrease_num=1, verbose=False, **kwargs):

        super().__init__(**kwargs)

        self.horizon = horizon
        self.num_sim_traj = num_simulated_trajectories
        self.factor_decrease_num = factor_decrease_num

        if num_simulated_trajectories < 2:
            raise ValueError("At least two trajectories needed!")

        self.verbose = verbose
        self.visualize_env = None
        self.model_dir = allogger.get_logger('root').logdir
        self.forward_model_state = None

    # checks if the model (in case of a ground truth model is consistent with the actual environment
    def check_model_consistency(self):
        if isinstance(self.forward_model, AbstractGroundTruthModel) and isinstance(self.env, GroundTruthSupportEnv):
            model_state = self.forward_model_state
            env_state = self.env.get_GT_state()
            diff = self.env.compute_state_difference(model_state, env_state)
            if diff > 1e-5:
                print(f"Warning: internal GT model and actual env are not in sync: Difference: {diff}")
                print("env state:", env_state)
                print("model_state:", model_state)

    @abstractmethod
    def sample_action_sequences(self, obs, num_traj, time_slice=None):
        """
        should return num_traj sampled trajectory of length self.horizon (or time_slice of it)
        """
        pass

    def simulate_trajectories(self, *, obs, state, action_sequences: np.ndarray) -> RolloutBuffer:
        """
        :param obs: current starting observation
        :param state: current starting state of forward model
        :param action_sequences: shape: [p,h,d]
        """
        num_parallel_trajs = action_sequences.shape[0]
        start_obs = np.array([obs] * num_parallel_trajs)  # shape:[p,d]
        start_states = [state] * num_parallel_trajs
        current_sim_policy = OpenLoopPolicy(action_sequences)
        return self.forward_model.predict_n_steps(start_observations=start_obs, start_states=start_states,
                                                  policy=current_sim_policy, horizon=self.horizon)[0]

    def beginning_of_rollout(self, *, observation, state=None, mode):
        if state is not None and isinstance(self.forward_model, AbstractGroundTruthModel):
            self.forward_model_state = state
        else:
            self.forward_model_state = self.forward_model.reset(observation)

    def save(self, data):
        if self.save_data:
            print("Saving controller data to ", self.save_data_to)
            with open(self.save_data_to, 'wb') as f:
                pickle.dump(data, f)

    def _create_path_to_file(self, to_file: str):
        self.save_data = True
        self.save_data_to = os.path.join(self.model_dir, to_file)


class MpcRandom(MpcController, RndController):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_action = self.env.action_space.sample()
        self.action_change_frequency = self.action_sampler_params.action_change_frequency
        assert self.action_change_frequency < self.horizon
        self.counter = 0

    def sample(self):
        if self.counter < self.action_change_frequency:
            self.counter += 1
        else:
            self.current_action = self.env.action_space.sample()
            self.counter = 0
        return self.current_action

    def sample_action_sequences(self, obs, num_traj, time_slice=None):

        samples = np.array([[self.sample() for _ in range(self.horizon)] for _ in range(num_traj)])
        # if samples.ndim == 2:
        #    samples = np.expand_dims(samples, axis=-1)
        return samples

    def prepare_action_sequences(self, *, num_traj):
        return self.sample_action_sequences(obs=None, num_traj=num_traj)  # shape:[p,h,d]

    def get_action(self, obs, state, mode="train"):
        if self.verbose and mode != "expert":  # in expert mode we are relabeling states that are not currently in env
            self.check_model_consistency()

        self.forward_model_state = self.forward_model.got_actual_observation_and_env_state(
            observation=obs, env_state=state, model_state=self.forward_model_state)

        action_sequences = self.prepare_action_sequences(num_traj=self.num_sim_traj)
        simulated_paths = self.simulate_trajectories(obs=obs, state=self.forward_model_state,
                                                     action_sequences=action_sequences)
        costs = self.trajectory_cost_fn(self.cost_fn, simulated_paths)  # shape: [num_sim_paths]

        best_traj_idx = np.argmin(costs)
        executed_action = simulated_paths[best_traj_idx]["actions"][0]

        if self.do_visualize_plan:
            obs = simulated_paths[best_traj_idx]["observations"]
            acts = simulated_paths[best_traj_idx]["actions"]
            self.visualize_plan(obs=obs, state=state, acts=acts)

        # for stateful models, actually simulate step (forward model stores the state internally)
        if self.forward_model_state is not None:
            obs_, self.forward_model_state, rewards = \
                self.forward_model.predict(observations=obs, states=self.forward_model_state,
                                           actions=executed_action)
        return executed_action


# standard implementation of a CEM
class MpcCemStd(MpcController):
    lower: Union[None, float, np.ndarray]
    upper: Union[None, float, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    model_evals_per_timestep: int
    elite_samples: RolloutBuffer

    def __init__(self, *, action_sampler_params, **kwargs):
        super().__init__(**kwargs)
        self._parse_action_sampler_params(**action_sampler_params)
        self._check_validity_parameters()

        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        self.was_reset = False

    def beginning_of_rollout(self, *, observation, state=None, mode):
        super().beginning_of_rollout(observation=observation, state=state, mode=mode)
        self.mean = self.get_init_mean(True)
        self.std = self.get_init_std(True)
        self.lower, self.upper = None, None
        self.elite_samples = RolloutBuffer()
        self.was_reset = True

        self._update_bounds(like_levine=self.like_levine)
        self.model_evals_per_timestep = self.num_sim_traj * self.opt_iter * self.horizon

        print(f"CEM-Standard using {self.model_evals_per_timestep} evaluations per step "
              f"and {self.model_evals_per_timestep / self.horizon} trajectories per step")

    def end_of_rollout(self, total_time, total_return, mode):
        super().end_of_rollout(total_time, total_return, mode)

    def get_init_mean(self, relative):
        if relative:
            return np.zeros(self.dim_samples) + (self.env.action_space.high + self.env.action_space.low) / 2.0
        else:
            return np.zeros(self.dim_samples)

    def get_init_std(self, relative):
        if relative:
            return np.ones(self.dim_samples) * \
                   (self.env.action_space.high - self.env.action_space.low) / 2.0 * self.init_std
        else:
            return self.init_std * np.ones(self.dim_samples)

    def sample_action_sequences(self, obs, num_traj, time_slice=None):
        """
        :param num_traj: number of trajectories
        :param obs: current observation
        :type time_slice: slice (not used here)
        """
        m = self.mean[None, :]
        s = self.std[None, :]
        samples = truncnorm.rvs(self.lower, self.upper, loc=m, scale=s,
                                size=(num_traj, *self.mean.shape))
        return samples  # shape:[p,h,d]

    def get_action(self, obs, state, mode="train"):
        simulated_paths = RolloutBuffer()

        if not self.was_reset:
            raise AttributeError("beginning_of_rollout() needs to be called before")

        if self.verbose:
            print(f"-------------------- {self.mean[0][0:6]}")

        def display_cost(cost):
            return cost / self.horizon if self.cost_along_trajectory == "sum" else cost

        self.forward_model_state = self.forward_model.got_actual_observation_and_env_state(
            observation=obs, env_state=state, model_state=self.forward_model_state)

        best_traj_idx = None
        costs = [float("inf")]
        num_sim_traj = self.num_sim_traj
        for i in range(self.opt_iter):
            action_sequences = self.sample_action_sequences(obs=obs, num_traj=num_sim_traj)
            simulated_paths = self.simulate_trajectories(obs=obs, state=self.forward_model_state,
                                                         action_sequences=action_sequences)

            costs = self.trajectory_cost_fn(self.cost_fn, simulated_paths)  # shape: [num_sim_paths]
            best_traj_idx = np.argmin(costs)

            if self.verbose:
                best_actions = simulated_paths[best_traj_idx]["actions"][0]
                print('iter {}:{} --- best cost: {:.2f} --- mean: {:.2f} --- worst: {:.2f}  best action: {}...'
                      .format(i, num_sim_traj, display_cost(np.amin(costs)),
                              display_cost(np.mean(costs)), display_cost(np.amax(costs)), best_actions[0:6]))
            self.update_distributions(simulated_paths, costs)
            # end of inner loop

        if self.execute_best_elite:
            executed_action = simulated_paths[best_traj_idx]["actions"][0]
        else:
            executed_action = self.mean[0].copy()

        # Shift mean time-wise
        if self.shift_means:
            self.mean[:-1] = self.mean[1:]
            last_predicted_ob = simulated_paths[best_traj_idx]["observations"][-1]
            self.mean[-1] = self.compute_new_mean(obs=last_predicted_ob)
        else:
            self.mean = np.zeros(self.dim_samples)

        # reset variance
        self.std = self.get_init_std(True)
        self._update_bounds(like_levine=self.like_levine)

        self.logger.log(display_cost(min(costs)), key="Expected_trajectory_cost")
        # self.logger.log(min(costs) / self.horizon, key="Expected average cost")

        if self.do_visualize_plan:
            viz_obs = simulated_paths[best_traj_idx]["observations"]
            acts = simulated_paths[best_traj_idx]["actions"]
            self.visualize_plan(obs=viz_obs, state=self.forward_model_state, acts=acts)

        # for stateful models, actually simulate step (forward model stores the state internally)
        if self.forward_model_state is not None:
            obs_, self.forward_model_state, rewards = \
                self.forward_model.predict(observations=obs, states=self.forward_model_state, actions=executed_action)
        return executed_action

    def compute_new_mean(self, obs):
        if self.like_levine:
            return self.mean[-1] * 0
        else:
            return self.mean[-1]

    def update_distributions(self, sampled_trajectories: RolloutBuffer, costs):
        """
        :param sampled_trajectories:
        :param costs: array of costs: shape (number trajectories)
        """
        # Get elite parameters
        elite_idxs = np.array(costs).argsort()[: self.num_elites]
        self.elite_samples = RolloutBuffer(rollouts=sampled_trajectories[elite_idxs])

        # Update mean, std
        elite_sequences = self.elite_samples.as_array("actions")

        # fit around mean of elites
        new_mean = elite_sequences.mean(axis=0)
        new_std = elite_sequences.std(axis=0)
        self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean  # [h,d]
        self.std = (1 - self.alpha) * new_std + self.alpha * self.std
        self._update_bounds(like_levine=self.like_levine)

    def _update_bounds(self, like_levine):
        if like_levine:
            lb, ub = self.env.action_space.low, self.env.action_space.high
            lb_dist, ub_dist = self.mean - lb, ub - self.mean
            self.std = np.maximum(1e-8, np.minimum(np.minimum(lb_dist / 2, ub_dist / 2), self.std))
            self.lower = -2
            self.upper = 2
        else:
            # the bounds are refering to a std-normal distribution.
            # bounds are such that we cannot leave the action space
            self.lower = (self.env.action_space.low - self.mean) / (self.std + 1e-8)
            self.upper = (self.env.action_space.high - self.mean) / (self.std + 1e-8)

    def _parse_action_sampler_params(self, *, alpha, elites_size, opt_iterations,
                                     init_std, shift_means, execute_best_elite, bounds_like_levine):

        self.alpha = alpha
        self.elites_size = elites_size
        self.opt_iter = opt_iterations
        self.init_std = init_std
        self.execute_best_elite = execute_best_elite
        self.like_levine = bounds_like_levine
        self.shift_means = shift_means

    def _check_validity_parameters(self):

        self.num_elites = min(self.elites_size, self.num_sim_traj // 2)
        if self.num_elites < 2:
            warn('Number of trajectories is too low for given elites_frac. Setting num_elites to 2.')
            self.num_elites = 2

        if isinstance(self.env.action_space, spaces.Discrete):
            # self.dim_samples = (self.horizon, self.env.action_space.n)
            raise NotImplementedError("CEM ERROR: Implement categorical distribution for discrete envs.")
        elif isinstance(self.env.action_space, spaces.Box):
            self.dim_samples = (self.horizon, self.env.action_space.shape[0])
        else:
            raise NotImplementedError
