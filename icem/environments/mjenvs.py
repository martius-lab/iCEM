from gym.utils import EzPickle

from .abstract_environments import *
from .mj_envs.mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0, ADD_BONUS_REWARDS as DoorEnvV0_ADD_BONUS_REWARDS
from .mj_envs.mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0, ADD_BONUS_REWARDS as RelocateEnvV0_ADD_BONUS_REWARDS
from .mj_envs.mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0, ADD_BONUS_REWARDS
from .mujoco import MujocoGroundTruthSupportEnv


class Door(MujocoGroundTruthSupportEnv, GoalSpaceEnvironmentInterface, DoorEnvV0):
    def __init__(self, *, name, frame_skip=None, add_bonus_rewards=True, shaped_reward=True,
                 use_normalized_actions=False, **kwargs):

        self.shaped_reward = shaped_reward
        self.use_normalized_actions = use_normalized_actions

        DoorEnvV0.__init__(self)
        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        GoalSpaceEnvironmentInterface.__init__(self)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name,
                          **self.init_kwargs)  # needed to call make the pickling work with the args given
        if frame_skip:
            self.frame_skip = frame_skip

        start_idx = self.data.qpos.ravel().shape[0] - 3 + 1 # qp[1:-2] + [latch_pos]
        self.door_pos_idx = np.arange(start_idx, start_idx + self.data.qpos[self.door_hinge_did].ravel().shape[0])
        self.palm_pos_idx = np.arange(self.door_pos_idx[-1]+1, self.door_pos_idx[-1]+1 + self.data.site_xpos[self.grasp_sid].ravel().shape[0])
        self.handle_pos_idx = np.arange(self.palm_pos_idx[-1]+1, self.palm_pos_idx[-1]+1 + self.data.site_xpos[self.handle_sid].ravel().shape[0])
        self.qv_start_idx = self.data.qvel.ravel().shape[0]

        self.add_bonus_rewards = add_bonus_rewards
        global DoorEnvV0_ADD_BONUS_REWARDS
        DoorEnvV0_ADD_BONUS_REWARDS = self.add_bonus_rewards

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        if self.use_normalized_actions:
            low[:] = -1
            high[:] = 1
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def goal_from_observation(self, observations):
        pass

    def achieved_goal_from_observation(self, observations):
        pass

    def overwrite_goal_inplace(self, observations, goals):
        pass

    def is_success(self, observation, action, next_obs):
        return next_obs[..., self.door_pos_idx] >= 1.35

    def cost_fn(self, observations, actions, next_observations):
        handle_pos = observations[..., self.handle_pos_idx]
        palm_pos = observations[..., self.palm_pos_idx]
        door_pos = observations[..., self.door_pos_idx].flatten()

        # get to handle
        if self.shaped_reward:
            cost = +0.1*np.linalg.norm(palm_pos-handle_pos, axis=-1)
        else:
            cost = 0
        # open door
        cost += +0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        cost += +1e-5*np.sum(observations[..., -self.qv_start_idx:]**2, axis=-1)

        if self.add_bonus_rewards:
            # Bonus
            cost -= 2 * (door_pos > 0.2).astype(np.int32)
            cost -= 8 * (door_pos > 1.0).astype(np.int32)
            cost -= 10 * (door_pos > 1.35).astype(np.int32)

        return cost

    def set_state_from_observation(self, observation):
        pass

    # def get_obs(self):
    #     obs = super().get_obs()
    #     qv = self.data.qvel.ravel()
    #
    #     return np.concatenate([obs, qv])

    def viewer_setup(self):
        self.viewer._hide_overlay = True

    # noinspection PyPep8Naming
    def set_GT_state(self, state):
        orig_state = state[:-3]
        door_pos = state[-3:]
        self.sim.set_state_from_flattened(orig_state.copy())
        self.model.body_pos[self.door_bid] = door_pos.copy()
        self.sim.forward()

    # noinspection PyPep8Naming
    def get_GT_state(self):
        return np.concatenate([self.sim.get_state().flatten(), self.model.body_pos[self.door_bid].ravel()])

    def step(self, a):
        if not self.use_normalized_actions:
            try:
                a = (a - self.act_mid) / self.act_rng
            except: 
                a = a

        return super().step(a)

class Relocate(MujocoGroundTruthSupportEnv, GoalSpaceEnvironmentInterface, RelocateEnvV0):
    def __init__(self, *, name, frame_skip=None, add_bonus_rewards=True, use_normalized_actions=False, **kwargs):
        self.use_normalized_actions = use_normalized_actions
        RelocateEnvV0.__init__(self)
        MujocoGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        GoalSpaceEnvironmentInterface.__init__(self)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name,
                          **self.init_kwargs)  # needed to call make the pickling work with the args given
        if frame_skip:
            self.frame_skip = frame_skip

        start_idx = self.data.qpos.ravel().shape[0] - 6 # qp[:-6]
        self.palm_pos_minus_obj_pos_idx = np.arange(start_idx, start_idx + 3)
        self.palm_pos_minus_target_pos_idx = np.arange(self.palm_pos_minus_obj_pos_idx[-1] + 1, self.palm_pos_minus_obj_pos_idx[-1] + 1 + 3)
        self.obj_pos_minus_target_pos_idx = np.arange(self.palm_pos_minus_target_pos_idx[-1] + 1, self.palm_pos_minus_target_pos_idx[-1] + 1 + 3)

        self.add_bonus_rewards = add_bonus_rewards
        global RelocateEnvV0_ADD_BONUS_REWARDS
        RelocateEnvV0_ADD_BONUS_REWARDS = self.add_bonus_rewards

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        if self.use_normalized_actions:
            low[:] = -1
            high[:] = 1
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def goal_from_observation(self, observations):
        pass

    def achieved_goal_from_observation(self, observations):
        pass

    def overwrite_goal_inplace(self, observations, goals):
        pass

    def is_success(self, observation, action, next_obs):
        return np.linalg.norm(next_obs[..., self.obj_pos_minus_target_pos_idx], axis=-1) < 0.1

    def cost_fn(self, observations, actions, next_observations):
        obj_pos = observations[..., -3:]
        palm_pos_minus_obj_pos = observations[..., self.palm_pos_minus_obj_pos_idx]
        palm_pos_minus_target_pos = observations[..., self.palm_pos_minus_target_pos_idx]
        obj_pos_minus_target_pos = observations[..., self.obj_pos_minus_target_pos_idx]

        cost = 0.1*np.linalg.norm(palm_pos_minus_obj_pos, axis=-1)  # take hand to object
        cost += -1.0 * (obj_pos[..., 2] > 0.04).astype(np.int32) # bonus for lifting the object
        # make hand go to target
        # cost += +0.5*np.linalg.norm(palm_pos_minus_target_pos, axis=-1) * (obj_pos[..., 2] > 0.04).astype(np.int32)
        # make object go to target
        cost += +0.5*np.linalg.norm(obj_pos_minus_target_pos, axis=-1) * (obj_pos[..., 2] > 0.04).astype(np.int32)

        if self.add_bonus_rewards:
            # bonus for object close to target
            cost += -10.0 * (np.linalg.norm(obj_pos_minus_target_pos, axis=-1) < 0.1).astype(np.int32)
            # bonus for object "very" close to target
            cost += -20.0 * (np.linalg.norm(obj_pos_minus_target_pos, axis=-1) < 0.05).astype(np.int32)

        return cost

    def set_state_from_observation(self, observation):
        pass

    def get_obs(self):
        obs = super().get_obs()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        return np.concatenate([obs, obj_pos])

    def viewer_setup(self):
        self.viewer.cam.azimuth = 25
        self.viewer.cam.elevation = -25
        self.viewer._hide_overlay = True
        self.viewer.cam.distance = 1.7

    # noinspection PyPep8Naming
    def set_GT_state(self, state):
        orig_state = state[:-5]
        obj_pos = state[-5:-3]
        target_obj_pos = state[-3:]
        self.sim.set_state_from_flattened(orig_state.copy())
        self.model.site_pos[self.target_obj_sid] = target_obj_pos.copy()
        self.model.body_pos[self.obj_bid,:2] = obj_pos.copy()
        self.sim.forward()

    # noinspection PyPep8Naming
    def get_GT_state(self):
        return np.concatenate([self.sim.get_state().flatten(), self.model.body_pos[self.obj_bid, :2].ravel(), self.model.site_pos[self.target_obj_sid].ravel()])

    def step(self, a):
        if not self.use_normalized_actions:
            a = (a - self.act_mid) / self.act_rng

        return super().step(a)

