import gym.envs.classic_control as gym_classic
import gym.envs.box2d as gym_box

from .abstract_environments import *
from helpers import sin_and_cos_to_radians


class DiscreteActionMountainCar(GroundTruthSupportEnv, DiscreteActionReshaper, gym_classic.MountainCarEnv):
    goal_state = np.array([[0.5, 0.0]])
    goal_mask = np.array([[1.0, 0.0]])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setattr(self.action_space, "low", self.low)
        setattr(self.action_space, "high", self.high)

    def get_GT_state(self):
        return self.state

    def set_GT_state(self, state):
        # noinspection PyAttributeOutsideInit
        self.state = state

    def set_state_from_observation(self, observation):
        self.set_GT_state(observation)


class DiscreteActionCartPole(DiscreteActionReshaper, gym_classic.CartPoleEnv):
    goal_state = np.array([[0.0, 0.0, 0.0, 0.0]])
    goal_mask = np.array([[1.0, 1.0, 1.0, 1.0]])


class ContinuousMountainCar(GroundTruthSupportEnv, gym_classic.Continuous_MountainCarEnv):
    goal_state = np.array([[0.5, 0.0]])
    goal_mask = np.array([[1.0, 0.0]])

    def get_GT_state(self):
        return self.state

    def set_GT_state(self, state):
        # noinspection PyAttributeOutsideInit
        self.state = state

    def set_state_from_observation(self, observation):
        self.set_GT_state(observation)


class ContinuousLunarLander(EnvWithDefaults, gym_box.LunarLanderContinuous):
    goal_state = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1]])
    goal_mask = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])


class ContinuousPendulum(GroundTruthSupportEnv, gym_classic.PendulumEnv):
    def set_GT_state(self, state):
        cos_theta, sin_theta, theta_dot = state
        theta = sin_and_cos_to_radians(sin_theta, cos_theta)
        # noinspection PyAttributeOutsideInit
        self.state = theta, theta_dot

    def get_GT_state(self):
        return self.state

    def set_state_from_observation(self, observation):
        self.set_GT_state(observation)

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def cost_fn(self, observation, action, next_obs):
        cos_theta, sin_theta, th_dot = observation.T
        th = sin_and_cos_to_radians(sin_theta, cos_theta)
        costs = self.angle_normalize(th) ** 2 + 0.1 * th_dot ** 2 + 0.001 * (np.squeeze(action) ** 2)

        return costs
