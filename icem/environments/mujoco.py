import math
from gym.envs.mujoco import ReacherEnv, MujocoEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv as HalfCheetah_v3
from gym.envs.mujoco.ant_v3 import AntEnv as Ant_v3
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv as Humanoid_v3
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv as HumanoidStandup_v2
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.utils import EzPickle
from math import atan2
from mujoco_py.generated import const

from .abstract_environments import *


class MujocoEnvWithDefaults(MujocoEnv, ABC):

    def viewer_setup(self):
        super().viewer_setup()
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.type = const.CAMERA_FIXED

    def get_fps(self):
        return 1.0/self.dt


class MujocoGroundTruthSupportEnv(GroundTruthSupportEnv, MujocoEnvWithDefaults, ABC):
    """ adds generic state operations for all Mujoco-based envs """
    window_exists = False

    # noinspection PyPep8Naming
    def set_GT_state(self, state):
        self.sim.set_state_from_flattened(state.copy())
        self.sim.forward()

    # noinspection PyPep8Naming
    def get_GT_state(self):
        return self.sim.get_state().flatten()

    # noinspection PyMethodMayBeStatic
    def prepare_for_recording(self):
        if not self.window_exists:
            from mujoco_py import GlfwContext
            GlfwContext(offscreen=True)
            self.window_exists = True


class HalfCheetahMaybeWithPosition(MujocoGroundTruthSupportEnv, HalfCheetah_v3):
    def __init__(self, *, name, frame_skip=None, penalise_flipping=False, **kwargs):
        HalfCheetah_v3.__init__(self, **kwargs)
        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name, **self.init_kwargs)
        # needed to call make the pickling work with the args given

        if frame_skip:
            self.frame_skip = frame_skip
        self.penalise_flipping = penalise_flipping

    def set_state_from_observation(self, observation):
        if self.observation_space.shape[0] != 18:
            raise AttributeError("For GT model use , set 'exclude_current_positions_from_observation': false")

        qpos, qvel = observation[: self.model.nq], observation[self.model.nq:]
        self.set_state(qpos, qvel)

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
        scores -= velocity

        if is_single_state:
            scores = scores[0]

        return scores  # [batch_size, models]

    def from_full_state_to_transformed_state(self, full_state):
        # return full_state[:, 1:]
        qpos, qvel = full_state[..., 1: self.model.nq], full_state[..., self.model.nq:]
        qpos = np.insert(qpos, 0, 0, axis=-1)
        # qpos, qvel = full_state[:, 1: self.model.nq], full_state[:, self.model.nq + 1:]
        next_position = qpos + self.dt * qvel
        transformed_state = np.concatenate((qpos, next_position), axis=-1)
        return transformed_state

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }
        return observation, reward, done, info


class Ant(MujocoGroundTruthSupportEnv, Ant_v3):
    def __init__(self, *, name, **kwargs):
        Ant_v3.__init__(self, **kwargs)
        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name,
                          **self.init_kwargs)  # needed to call make the pickling work with the args given

    def set_state_from_observation(self, observation):
        q_pos, q_vel = observation[: self.model.nq], observation[self.model.nq: self.model.nq + self.model.nv]
        self.set_state(q_pos, q_vel)

    def are_states_unhealthy(self, states):
        min_z, max_z = self._healthy_z_range
        is_unhealthy = 1 - np.isfinite(states).all(axis=-1) * (min_z <= states[..., 2]) * (states[..., 2] <= max_z)
        return is_unhealthy

    def cost_fn(self, observation, action, next_obs):
        is_single_state = len(observation.shape) == 1
        if is_single_state and observation.shape != 113 or not is_single_state and observation.shape[-1] != 113:
            raise AttributeError(
                "If you wanna use this cost function, set " "'exclude_current_positions_from_observation': false"
            )

        if is_single_state:
            observation = observation[None, ...]
            action = action[None, ...]
            next_obs = next_obs[None, ...]

        unhealthy = self.are_states_unhealthy(observation)
        x_velocity = (next_obs[..., 0] - observation[..., 0]) / self.dt
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action), axis=-1)
        scores = -x_velocity + 100 * unhealthy + control_cost

        if is_single_state:
            scores = scores[0]

        return scores


class Hopper(MujocoGroundTruthSupportEnv, HopperEnv):

    def __init__(self, *, name, frame_skip=None, **kwargs):
        HopperEnv.__init__(self)
        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name, **self.init_kwargs)
        if frame_skip:
            self.frame_skip = frame_skip

    def set_state_from_observation(self, observation):
        q_pos, q_vel = observation[: self.model.nq], observation[self.model.nq: self.model.nq + self.model.nv]
        self.set_state(q_pos, q_vel)

    def unhealthy_states(self, states):
        z = states[..., 1]
        angle = states[..., 2]
        state = states[..., 2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=-1)
        healthy_z = (min_z < z) * (z < max_z)
        healthy_angle = (min_angle < angle) * (angle < max_angle)
        is_healthy = np.logical_and(healthy_state, healthy_z, healthy_angle)

        is_unhealthy = 1 - np.isfinite(states).all(axis=-1) * is_healthy
        return is_unhealthy

    def cost_fn(self, observation, action, next_obs):
        is_single_state = len(observation.shape) == 1
        if is_single_state and observation.shape != 12 or not is_single_state and observation.shape[-1] != 12:
            raise AttributeError(
                "If you wanna use this cost function, set " "'exclude_current_positions_from_observation': false",
                observation.shape
            )

        if is_single_state:
            observation = observation[None, ...]
            action = action[None, ...]
            next_obs = next_obs[None, ...]

        x_velocity = (next_obs[..., 0] - observation[..., 0]) / self.dt
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action), axis=-1)
        unhealthy = self.unhealthy_states(observation)
        scores = - x_velocity + 200 * unhealthy + control_cost

        if is_single_state:
            scores = scores[0]
        return scores


class HumanoidStandup(MujocoGroundTruthSupportEnv, HumanoidStandup_v2):
    def __init__(self, *, name, **kwargs):
        HumanoidStandup_v2.__init__(self)
        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name, **self.init_kwargs)
        # needed to call make the pickling work with the args given

    def viewer_setup(self):
        super().viewer_setup()
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.type = const.CAMERA_FIXED

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat,
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def set_state_from_observation(self, observation):
        q_pos = observation[: self.model.nq]
        q_vel = observation[self.model.nq: self.model.nq + self.model.nv]
        self.set_state(q_pos, q_vel)

    def cost_fn(self, observation, action, next_obs):
        is_single_state = len(observation.shape) == 1

        if is_single_state:
            observation = observation[None, ...]
            action = action[None, ...]
            # next_state = next_state[None, ...]

        pos = observation[..., 2]
        up = pos

        ctrl_cost = 0.1 * np.square(action).sum(axis=-1)

        scores = -up + ctrl_cost

        if is_single_state:
            scores = scores[0]

        return scores


class Humanoid(MujocoGroundTruthSupportEnv, Humanoid_v3):
    def __init__(self, *, name, **kwargs):
        Humanoid_v3.__init__(self, **kwargs)
        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name, **self.init_kwargs)
        # needed to call make the pickling work with the args given

    def viewer_setup(self):
        super().viewer_setup()
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.type = const.CAMERA_FIXED

    def set_state_from_observation(self, observation):
        q_pos = observation[: self.model.nq]
        q_vel = observation[self.model.nq: self.model.nq + self.model.nv]
        self.set_state(q_pos, q_vel)

    def are_states_unhealthy(self, states):
        min_z, max_z = self._healthy_z_range
        is_unhealthy = 1 - np.isfinite(states).all(axis=-1) * (min_z <= states[..., 2]) * (states[..., 2] <= max_z)
        return is_unhealthy

    def unhealthy_states(self, states):
        if self._exclude_current_positions_from_observation:
            z = states[..., 0]
        else:
            z = states[..., 2]
        min_z, max_z = self._healthy_z_range

        healthy_z = (min_z < z) * (z < max_z)
        is_healthy = healthy_z

        is_unhealthy = 1 - np.isfinite(states).all(axis=-1) * is_healthy

        return is_unhealthy

    def cost_fn(self, observation, action, next_obs):
        is_single_state = len(observation.shape) == 1

        # if is_single_state and observation.shape != 378 or not is_single_state and observation.shape[-1] != 378:
        #     raise AttributeError(
        #         "If you wanna use this cost function, set " "'exclude_current_positions_from_observation': false"
        #     )

        if is_single_state:
            observation = observation[None, ...]
            action = action[None, ...]
            next_obs = next_obs[None, ...]

        unhealthy = self.unhealthy_states(observation)
        penalty = 100  # self._healthy_reward
        if self._exclude_current_positions_from_observation:
            x_velocity = observation[..., self.model.nq-2]
        else:
            x_velocity = observation[..., self.model.nq]

        control_cost = self._ctrl_cost_weight * np.sum(np.square(action), axis=-1)
        scores = -self._forward_reward_weight * x_velocity + penalty * unhealthy + control_cost

        if is_single_state:
            scores = scores[0]

        return scores


class Reacher(MujocoGroundTruthSupportEnv, ReacherEnv):
    def __init__(self, *, name, frame_skip=None, **kwargs):
        ReacherEnv.__init__(self)
        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name, **self.init_kwargs)
        # needed to call make the pickling work with the args given
        if frame_skip:
            self.frame_skip = frame_skip

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def set_state_from_observation(self, observation):
        theta1 = atan2(observation[2], observation[0])
        theta2 = atan2(observation[3], observation[1])
        q_pos = np.concatenate(([theta1, theta2], observation[4:6]))
        q_vel = np.concatenate((observation[6:8], [0.0, 0.0]))
        self.set_state(q_pos, q_vel)

    def cost_fn(self, observations, actions, next_observations):
        fingertip_target_diff = observations[..., -3:]
        return np.linalg.norm(fingertip_target_diff, axis=-1)  # + (np.sum(actions ** 2, axis=-1))
