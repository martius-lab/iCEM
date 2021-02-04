import os
import pickle
from warnings import warn

import allogger
import colorednoise
import numpy as np
from gym import spaces
from scipy.stats import truncnorm

from controllers.mpc import MpcController
from misc.rolloutbuffer import RolloutBuffer


# our improved CEM
class MpcICem(MpcController):
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
        self.elite_samples = RolloutBuffer()
        self.was_reset = True

        self.model_evals_per_timestep = \
            sum([max(self.elites_size * 2, int(self.num_sim_traj / (self.factor_decrease_num ** i)))
                 for i in range(0, self.opt_iter)]) * self.horizon

        print(f"iCEM using {self.model_evals_per_timestep} evaluations per step "
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
        :type time_slice: slice
        """
        # colored noise
        if self.noise_beta > 0:
            assert (self.mean.ndim == 2)
            # Important improvement
            # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
            # noinspection PyUnresolvedReferences
            samples = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=(num_traj, self.mean.shape[1],
                                                                                self.mean.shape[0])).transpose(
                [0, 2, 1])
        else:
            samples = np.random.randn(num_traj, *self.mean.shape)

        samples = np.clip(samples * self.std + self.mean, self.env.action_space.low, self.env.action_space.high)
        if time_slice is not None:
            samples = samples[:, time_slice]
        return samples

    def prepare_action_sequences(self, *, obs, num_traj, iteration):
        sampled_from_distribution = self.sample_action_sequences(obs, num_traj)
        # shape:[p,h,d]
        if self.use_mean_actions and iteration == self.opt_iter - 1:
            sampled_from_distribution[0] = self.mean
        return sampled_from_distribution

    def elites_2_action_sequences(self, *, elites, obs, fraction_to_be_used=1.0):
        """
        :param obs: current observation of shape [obs_dim]
        :param fraction_to_be_used:
        :type elites: RolloutBuffer
        """
        actions = elites.as_array("actions")  # shape: [p,h,d]
        reused_actions = actions[:, 1:]  # shape: [p,h-1,d]
        num_elites = int(reused_actions.shape[0] * fraction_to_be_used)
        reused_actions = reused_actions[:num_elites]
        # shape:[p,1,d]
        last_actions = self.sample_action_sequences(time_slice=slice(-1, None), obs=obs, num_traj=num_elites)

        return np.concatenate([reused_actions, last_actions], axis=1)

    def get_action(self, obs, state, mode="train"):
        simulated_paths = RolloutBuffer()

        if not self.was_reset:
            raise AttributeError("beginning_of_rollout() needs to be called before")

        if self.verbose:
            print(f"-------------------- {self.mean[0][0:6]}")
            if mode != "expert":  # in expert mode we are relabeling states that are not currently in env
                self.check_model_consistency()

        self.forward_model_state = self.forward_model.got_actual_observation_and_env_state(
            observation=obs, env_state=state, model_state=self.forward_model_state)

        best_traj_idx = None
        costs = [float("inf")]

        num_sim_traj = self.num_sim_traj
        for i in range(self.opt_iter):
            # Decay of sample size
            if i > 0:  # Important improvement
                num_sim_traj = max(self.elites_size * 2, int(num_sim_traj / self.factor_decrease_num))

            action_sequences = self.prepare_action_sequences(obs=obs, num_traj=num_sim_traj, iteration=i)
            # Shifting elites over time: minor improvement?
            if i == 0 and self.shift_elites_over_time and self.elite_samples:
                action_seq_from_elites = self.elites_2_action_sequences(
                    elites=self.elite_samples, fraction_to_be_used=self.fraction_elites_reused, obs=obs
                )
                action_sequences = np.concatenate(
                    [action_sequences, action_seq_from_elites], axis=0
                )  # shape [p+pe,h,d]

            simulated_paths = self.simulate_trajectories(obs=obs, state=self.forward_model_state,
                                                         action_sequences=action_sequences)

            # keep elites from prev. iteration  # Important improvement
            if i > 0 and self.keep_previous_elites:
                assert self.elite_samples
                simulated_paths.extend(self.elite_samples[:int(len(self.elite_samples) * self.fraction_elites_reused)])

            orig_cost = self.trajectory_cost_fn(self.cost_fn, simulated_paths)  # shape: [num_sim_paths]
            costs = orig_cost.copy()
            best_traj_idx = np.argmin(costs)

            if self.verbose:
                def display_cost(cost):
                    return cost / self.horizon if self.cost_along_trajectory == "sum" else cost

                best_actions = simulated_paths[best_traj_idx]["actions"][0]
                print('iter {}:{} --- best cost: {:.2f} --- mean: {:.2f} --- worst: {:.2f}  best action: {}...'
                      .format(i, num_sim_traj, display_cost(np.amin(costs)),
                              display_cost(np.mean(costs)), display_cost(np.amax(costs)), best_actions[0:6]))
            self.update_distributions(simulated_paths, costs)

            # end of inner loop

        executed_action = simulated_paths[best_traj_idx]["actions"][0]

        ### Shift initialization ###
        # Shift mean time-wise
        self.mean[:-1] = self.mean[1:]

        # compute new action (default is to preserve the last one)
        last_predicted_ob = simulated_paths[best_traj_idx]["observations"][-1]
        self.mean[-1] = self.compute_new_mean(obs=last_predicted_ob)
        ############################

        ### initialization of std dev ###
        self.std = self.get_init_std(True)

        self.logger.log(min(costs), key="Expected_trajectory_cost")
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
        return self.mean[-1]

    def update_distributions(self, sampled_trajectories: RolloutBuffer, costs):
        """
        :param sampled_trajectories:
        :param costs: array of costs: shape (number trajectories)
        """
        elite_idxs = np.array(costs).argsort()[: self.num_elites]

        self.elite_samples = RolloutBuffer(rollouts=sampled_trajectories[elite_idxs])

        # Update mean, std
        elite_sequences = self.elite_samples.as_array("actions")

        # fit around mean of elites
        new_mean = elite_sequences.mean(axis=0)
        new_std = elite_sequences.std(axis=0)

        self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean  # [h,d]
        self.std = (1 - self.alpha) * new_std + self.alpha * self.std

    def _parse_action_sampler_params(
            self, *,
            alpha,
            elites_size,
            opt_iterations,
            init_std,
            use_mean_actions,
            keep_previous_elites,
            shift_elites_over_time,
            fraction_elites_reused,
            noise_beta=1):

        self.alpha = alpha
        self.elites_size = elites_size
        self.opt_iter = opt_iterations
        self.init_std = init_std
        self.use_mean_actions = use_mean_actions
        self.keep_previous_elites = keep_previous_elites
        self.shift_elites_over_time = shift_elites_over_time
        self.fraction_elites_reused = fraction_elites_reused
        self.noise_beta = noise_beta

    def _check_validity_parameters(self):

        self.num_elites = min(self.elites_size, self.num_sim_traj // 2)
        if self.num_elites < 2:
            warn('Number of trajectories is too low for given elites_frac. Setting num_elites to 2.')
            self.num_elites = 2

        if isinstance(self.env.action_space, spaces.Discrete):
            raise NotImplementedError("CEM ERROR: Implement categorical distribution for discrete envs.")
        elif isinstance(self.env.action_space, spaces.Box):
            self.dim_samples = (self.horizon, self.env.action_space.shape[0])
        else:
            raise NotImplementedError
