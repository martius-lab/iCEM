import time
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from misc.base_types import Controller, Env, ForwardModel
from environments import env_from_string
from environments.abstract_environments import GroundTruthSupportEnv
from misc.rolloutbuffer import RolloutBuffer
from .utils import ArrayIteratorParallelRowwise


class TrainableController(Controller, ABC):
    needs_training = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cost_fn = self.env.cost_fn
        self.reward_fn = self.env.reward_fn

    @abstractmethod
    def train(self, rollout_buffer: RolloutBuffer):
        """
        Trains the controller from experience
        """
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class TrainableFromTrajectoriesController(TrainableController, ABC):
    needs_data = True


class StatefulController(Controller, ABC):
    has_state = True

    @abstractmethod
    def beginning_of_rollout(self, *, observation, state=None, mode):
        """
        Runs before each rollout, can be used to reset the state of the controller
        """
        pass

    @abstractmethod
    def end_of_rollout(self, total_time, total_return, mode):
        """
        Runs at the end of each rollout, can be used, for instance, to store some data
        """
        pass


class ModelBasedController(Controller, ABC):
    forward_model: ForwardModel

    def __init__(self, *, forward_model, env: Env,  cost_along_trajectory, do_visualize_plan=None,
                 use_env_reward_as_cost=False, **kwargs):
        super().__init__(env=env, **kwargs)
        self.forward_model = forward_model
        self.do_visualize_plan = do_visualize_plan
        self.visualize_env = None
        self.cost_fn = self.env.cost_fn
        self.cost_along_trajectory = cost_along_trajectory
        self.use_env_reward_as_cost = use_env_reward_as_cost

    def trajectory_cost_fn(self, cost_fn, rollout_buffer: RolloutBuffer):
        if self.use_env_reward_as_cost:
            costs_path = -rollout_buffer.as_array('rewards')
        else:
            costs_path = np.asarray(
                [cost_fn(r["observations"], r["actions"], r["next_observations"]) for r in rollout_buffer]
            )  # shape: [p,h]

        if self.cost_along_trajectory == "sum":
            return np.sum(costs_path, axis=1)
        elif self.cost_along_trajectory == "best":
            return np.amin(np.array(costs_path), axis=1)
        elif self.cost_along_trajectory == "final":
            return np.asarray(costs_path)[:, -1]
        else:
            raise NotImplementedError(
                "Implement method {} to compute cost along trajectory".format(self.cost_along_trajectory)
            )

    def visualize_plan(self, *, obs, state, acts):
        """ visualizes a given plan. In mode 'last' it will render the end of the plan in a copy of the env.
        In mode 'all' it will render the entire plan. """
        if self.do_visualize_plan is None or not self.do_visualize_plan:
            pass
        if isinstance(self.env, GroundTruthSupportEnv) and self.env.supports_live_rendering:  # can't do it for dm suite
            if self.visualize_env is None:
                if self.env.enable_rendering_at_init:
                    init_kwargs = deepcopy(self.env.init_kwargs)
                    init_kwargs['enable_rendering'] = True
                    self.visualize_env = env_from_string(self.env.name, **init_kwargs)
                else:
                    self.visualize_env = env_from_string(self.env.name, **self.env.init_kwargs)
                self.visualize_env.reset()

            if self.do_visualize_plan == "last":
                self.visualize_env.set_state_from_observation(obs[-1])
                self.visualize_env.step(acts[-1])
                self.visualize_env.render()
            elif self.do_visualize_plan == "all":
                # print("State at visual ", state)
                self.visualize_env.set_GT_state(state)
                reported = False

                for i, a in enumerate(acts):
                    new_obs, *_ = self.visualize_env.step(a)
                    if i < len(obs) - 1 and np.linalg.norm(new_obs - obs[i + 1]) > 0.01 and not reported:
                        reported = True
                        print(f"simulation for visualization does not match mental model at {i}: ")
                        print(f"orig: ", obs[i + 1])
                        print(f"simu: ", new_obs)

                    self.visualize_env.render()
                    time.sleep(1.0 / 25.0)
            else:
                raise AttributeError("unknown mode for do_visualize_plan: Options: None, 'last','all'")


class ParallelController(Controller, ABC):
    @abstractmethod
    def get_parallel_policy_copy(self, indices):
        pass


class NeedsPolicyController(Controller, ABC):
    @abstractmethod
    def set_policy(self, policy):
        pass


class TeacherController(Controller, ABC):
    @abstractmethod
    def select_teacher_rollouts_for_training(self, rollouts):
        pass

    @abstractmethod
    def student_rollouts(self, rollouts):
        pass


class OpenLoopPolicy(ParallelController):
    def __init__(self, action_sequences: np.ndarray, *, env: Env = None):
        """:param action_sequences: shape: [p, h, d]
        """
        super().__init__(env=env)
        self.action_sequences = action_sequences
        self.action_sequence_iterator = None

    @staticmethod
    def get_num_parallel(obs):
        if obs.ndim == 1:
            return 1
        else:
            return obs.shape[0]

    def get_action(self, obs: np.ndarray, state, mode="train"):
        """ Every time get_action is called we take the actions from the actions_sequence and return it.
        In case we are asked to return fewer (parallel) actions then we are set up for (p above)
        then we first continue this amount of roll-outs and then proceed to the next sub-batch
        :param state:
        :param obs: shape [p, d] (number parallel runs, state-dim)
        :param mode: unused
        """

        if self.action_sequence_iterator is None:
            self.action_sequence_iterator = ArrayIteratorParallelRowwise(
                self.action_sequences, self.get_num_parallel(obs)
            )
        return self.action_sequence_iterator.__next__()

    def get_parallel_policy_copy(self, indices):
        return OpenLoopPolicy(self.action_sequences[indices], env=self.env)


class MpcHook(ABC):
    @abstractmethod
    def considered_trajectories(self, state, simulated_trajectories, costs, aux_costs, is_fine_tuned=False):
        pass

    @abstractmethod
    def executed_action(self, state, action):
        pass

    @abstractmethod
    def beginning_of_rollout(self, observation):
        pass

    @abstractmethod
    def end_of_rollout(self):
        pass
