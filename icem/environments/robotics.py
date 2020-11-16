from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv as FetchPickAndPlaceEnv_v1
from gym.envs.robotics.fetch.reach import FetchReachEnv
from gym.envs.robotics.robot_env import RobotEnv
from gym.utils import EzPickle

from .abstract_environments import *


class GymRoboticsGroundTruthSupportEnv(GroundTruthSupportEnv, RobotEnv, ABC):
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


class FetchPickAndPlace(MaskedGoalSpaceEnvironmentInterface, GymRoboticsGroundTruthSupportEnv, FetchPickAndPlaceEnv_v1):
    def __init__(self, *, name, sparse, threshold, fixed_object_pos=None, fixed_goal=None,
                 shaped_reward=False, **kwargs):

        self.fixed_object_pos = fixed_object_pos
        self.fixed_goal = fixed_goal
        self.shaped_reward = shaped_reward

        FetchPickAndPlaceEnv_v1.__init__(self, **kwargs)
        GymRoboticsGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name, sparse=sparse, threshold=threshold,
                          **kwargs)  # needed to call make the pickling work with the args given

        assert (isinstance(self.observation_space, spaces.Dict))
        orig_obs_len = self.observation_space.spaces['observation'].shape[0]
        goal_space_size = self.observation_space.spaces['desired_goal'].shape[0]

        goal_idx = np.arange(orig_obs_len, orig_obs_len + goal_space_size)

        achieved_goal_idx = [3, 4, 5]

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(orig_obs_len + goal_space_size,), dtype='float32')

        MaskedGoalSpaceEnvironmentInterface.__init__(self, name=name, goal_idx=goal_idx,
                                                     achieved_goal_idx=achieved_goal_idx, sparse=sparse,
                                                     threshold=threshold)

    def _step_callback(self):
        self.sim.forward()  # we need to call forward because part of the model was overwritten and it is not consistent

    def get_pos_vel_of_joints(self, names):
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            return (
                np.array([self.sim.data.get_joint_qpos(name) for name in names]),
                np.array([self.sim.data.get_joint_qvel(name) for name in names]),
            )

    def set_pos_vel_of_joints(self, names, q_pos, q_vel):
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            for n, p, v in zip(names, q_pos, q_vel):
                self.sim.data.set_joint_qpos(n, p)
                self.sim.data.set_joint_qvel(n, v)

    @staticmethod
    def flatten_observation(obs):
        return np.concatenate((obs['observation'], obs['desired_goal']))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.flatten_observation(obs), reward, done, info

    def reset(self):
        # return self.flatten_observation(super().reset())
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return self.flatten_observation(obs)

    def get_GT_state(self):
        return np.concatenate((super().get_GT_state(), self.goal))

    def set_GT_state(self, state):
        mj_state = state[:-3]
        self.goal = state[-3:]
        super().set_GT_state(mj_state)

    def set_state_from_observation(self, observation):
        raise NotImplementedError("FetchPickAndPlace env needs the real GT states to be reset")

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:

            if self.fixed_object_pos is not None:
                object_xpos = self.initial_gripper_xpos[:2] + np.asarray(self.fixed_object_pos) * self.obj_range
            else:
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            if self.fixed_goal is not None:
                goal = self.initial_gripper_xpos[:3] + np.asarray(self.fixed_goal) * self.target_range
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air:
                    goal[2] += self.fixed_goal[2] * 0.45
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                              size=3)
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)

        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)

        return goal.copy()

    def cost_fn(self, observation, action, next_obs):

        dist_box_to_goal = np.linalg.norm(self.goal_from_observation(observation) -
                                          self.achieved_goal_from_observation(observation), axis=-1)

        dist_end_eff_to_box = 0
        if self.shaped_reward:
            dist_end_eff_to_box = np.linalg.norm(observation[:, :3] - observation[:, 3:6], axis=-1)

        if self.sparse:
            cost = np.asarray(dist_box_to_goal > self.threshold, dtype=np.float32) + \
                   np.asarray(dist_end_eff_to_box > self.threshold, dtype=np.float32) * 0.1
        else:
            cost = dist_box_to_goal + dist_end_eff_to_box * 0.1
        return cost

    def is_success(self, observation, action, next_obs):

        dist = np.linalg.norm(self.goal_from_observation(next_obs) -
                              self.achieved_goal_from_observation(next_obs), axis=-1)

        is_success = np.asarray(dist <= self.threshold, dtype=np.float32)

        return is_success


class FetchReach(MaskedGoalSpaceEnvironmentInterface, GymRoboticsGroundTruthSupportEnv, FetchReachEnv):
    def __init__(self, *, name, sparse, threshold, fixed_goal=None,
                 **kwargs):

        self.fixed_goal = fixed_goal

        FetchReachEnv.__init__(self, **kwargs)
        GymRoboticsGroundTruthSupportEnv.__init__(self, name=name, **kwargs)
        self.store_init_arguments(locals())
        EzPickle.__init__(self, name=name, sparse=sparse, threshold=threshold,
                          **kwargs)  # needed to call make the pickling work with the args given

        assert (isinstance(self.observation_space, spaces.Dict))
        orig_obs_len = self.observation_space.spaces['observation'].shape[0]
        self.goal_space_size = self.observation_space.spaces['desired_goal'].shape[0]

        goal_idx = np.arange(orig_obs_len, orig_obs_len + self.goal_space_size)

        achieved_goal_idx = [0, 1, 2]

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(orig_obs_len + self.goal_space_size,),
                                            dtype='float32')

        MaskedGoalSpaceEnvironmentInterface.__init__(self, name=name, goal_idx=goal_idx,
                                                     achieved_goal_idx=achieved_goal_idx, sparse=sparse,
                                                     threshold=threshold)

    def _step_callback(self):
        self.sim.forward()  # we need to call forward because part of the model was overwritten and it is not consistent

    def get_pos_vel_of_joints(self, names):
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            return (
                np.array([self.sim.data.get_joint_qpos(name) for name in names]),
                np.array([self.sim.data.get_joint_qvel(name) for name in names]),
            )

    def set_pos_vel_of_joints(self, names, q_pos, q_vel):
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            for n, p, v in zip(names, q_pos, q_vel):
                self.sim.data.set_joint_qpos(n, p)
                self.sim.data.set_joint_qvel(n, v)

    @staticmethod
    def flatten_observation(obs):
        return np.concatenate((obs['observation'], obs['desired_goal']))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.flatten_observation(obs), reward, done, info

    def reset(self):
        return self.flatten_observation(super().reset())

    def get_GT_state(self):
        return np.concatenate((super().get_GT_state(), self.goal))

    def set_GT_state(self, state):
        mj_state = state[:-3]
        self.goal = state[-3:]
        super().set_GT_state(mj_state)

    def set_state_from_observation(self, observation):
        raise NotImplementedError("FetchPickAndPlace env needs the real GT states to be reset")

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:

            if self.fixed_object_pos is not None:
                object_xpos = self.initial_gripper_xpos[:2] + np.asarray(self.fixed_object_pos) * self.obj_range
            else:
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            if self.fixed_goal is not None:
                goal = self.initial_gripper_xpos[:3] + np.asarray(self.fixed_goal) * self.target_range
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air:
                    goal[2] += self.fixed_goal[2] * 0.45
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range,
                                                                              size=3)
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)

        else:
            if self.fixed_goal is not None:
                goal = self.initial_gripper_xpos[:3] + np.asarray(self.fixed_goal)
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)

        return goal.copy()

    def cost_fn(self, observation, action, next_obs):

        dist_gripper_to_goal = np.linalg.norm(self.goal_from_observation(observation) -
                                              self.achieved_goal_from_observation(observation), axis=-1)

        if self.sparse:
            cost = np.asarray(dist_gripper_to_goal > self.threshold, dtype=np.float32)
        else:
            cost = dist_gripper_to_goal
        return cost

    def is_success(self, observation, action, next_obs):

        dist = np.linalg.norm(self.goal_from_observation(next_obs) -
                              self.achieved_goal_from_observation(next_obs), axis=-1)

        is_success = np.asarray(dist <= self.threshold, dtype=np.float32)

        return is_success


if __name__ == '__main__':
    env = FetchPickAndPlace(name='blub', sparse=False, threshold=0.05, fixed_goal=[0.5, -0.3, 0.6],
                            fixed_object_pos=[0.85, 0.85])

    while True:
        env.reset()
        for _ in range(50):
            env.render()
            env.step(env.action_space.sample())
