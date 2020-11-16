import pyglet
from gym import spaces
from gym.utils import seeding
from dm_env import specs
import gym
import numpy as np
import sys
from dm_control import suite, viewer
from dm_control.viewer import util, runtime
from dm_control.rl.control import flatten_observation
from dm_env import _environment as environment

from abc import ABC

import glfw
from numpy.random.mtrand import RandomState

from environments.abstract_environments import GroundTruthSupportEnv


class MyRuntime(viewer.application.runtime.Runtime):
    def _start(self):
        return True


class MyApplication(viewer.application.Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pause_subject = util.ObservableFlag(False)

    def _tick(self):
        super()._tick()
        if self._runtime and self._runtime.state == runtime.State.STOPPED:
            # noinspection PyProtectedMember
            glfw.set_window_should_close(self._window._context.window, 1)


viewer.application.Application = MyApplication
viewer.application.runtime.Runtime = MyRuntime


class DmControlViewer(object):
    def __init__(self, width, height, depth=False):
        self.window = pyglet.window.Window(width=width, height=height, display=None)
        self.width = width
        self.height = height

        self.depth = depth

        if depth:
            self.format = "RGB"
            self.pitch = self.width * -3
        else:
            self.format = "RGB"
            self.pitch = self.width * -3

    def update(self, pixel):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        if self.depth:
            pixel = np.dstack([pixel.astype(np.uint8)] * 3)
        pyglet.image.ImageData(self.width, self.height, self.format, pixel.tobytes(), pitch=self.pitch).blit(0, 0)
        self.window.flip()

    def close(self):
        self.window.close()


class DmcDiscrete(gym.spaces.Discrete):
    def __init__(self, _minimum, _maximum):
        super().__init__(_maximum - _minimum)
        self.offset = _minimum


def convert_spec_to_space(spec, clip_inf=False):
    if spec.dtype == np.int:
        # Discrete
        return DmcDiscrete(spec.minimum, spec.maximum)
    else:
        # Box
        if type(spec) is specs.Array:
            return spaces.Box(-np.inf, np.inf, shape=spec.shape, dtype=np.float32)
        elif type(spec) is specs.BoundedArray:
            _min = spec.minimum
            _max = spec.maximum
            if clip_inf:
                _min = np.clip(spec.minimum, -sys.float_info.max, sys.float_info.max)
                _max = np.clip(spec.maximum, -sys.float_info.max, sys.float_info.max)

            if np.isscalar(_min) and np.isscalar(_max):
                # same min and max for every element
                return spaces.Box(_min, _max, shape=spec.shape, dtype=np.float32)
            else:
                # different min and max for every element
                return spaces.Box(_min + np.zeros(spec.shape), _max + np.zeros(spec.shape), dtype=np.float32)
        else:
            raise ValueError("Unknown spec!")


def convert_ordered_dict_to_space(odict):
    if len(odict.keys()) == 1:
        # no concatenation
        return convert_spec_to_space(list(odict.values())[0])
    else:
        # concatenation
        num_dim = sum([np.int(np.prod(odict[key].shape)) for key in odict])
        return spaces.Box(-np.inf, np.inf, shape=(num_dim,), dtype=np.float32)


def convert_observation(spec_obs):
    if len(spec_obs.keys()) == 1:
        # no concatenation
        return list(spec_obs.values())[0]
    else:
        # concatenation
        numdim = sum([np.int(np.prod(spec_obs[key].shape)) for key in spec_obs])
        space_obs = np.zeros((numdim,))
        i = 0
        for key in spec_obs:
            space_obs[i: i + np.prod(spec_obs[key].shape)] = spec_obs[key].ravel()
            i += np.prod(spec_obs[key].shape)
        return space_obs


class DmControlWrapper(GroundTruthSupportEnv, ABC):
    np_random: RandomState
    # Set goal_state and goal_mask for every env
    goal_state = None
    goal_mask = None
    domain_name = None
    supports_live_rendering = False

    def __init__(self, *, name, task_name, task_kwargs=None, visualize_reward=True, render_mode="human", **kwargs):

        super().__init__(name=name, **kwargs)
        self.store_init_arguments(locals())

        self.dmcenv = suite.load(
            domain_name=self.domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
        )

        # convert spec to space
        self.action_space = convert_spec_to_space(self.dmcenv.action_spec(), clip_inf=True)
        self.observation_space = convert_ordered_dict_to_space(self.dmcenv.observation_spec())

        self.metadata["render.modes"] = list(render_mode_list.keys())
        self.viewer = {key: None for key in render_mode_list.keys()}
        self.render_mode = render_mode
        self.render_mode_list = render_mode_list
        self.pixels = None
        self.timestep = None

        # set seed
        self.seed()

    # noinspection PyPep8Naming
    def set_GT_state(self, state):
        self.dmcenv._step_count = state[0]
        self.dmcenv._physics.set_state(state[1:])
        self.dmcenv._physics.after_reset()

        if self.timestep is not None:
            observation = self.dmcenv._task.get_observation(self.dmcenv._physics)
            if self.dmcenv._flat_observation:
                observation = flatten_observation(observation)

            self.timestep = environment.TimeStep(
                step_type=self.timestep.step_type,
                reward=None,
                discount=None,
                observation=observation)

    # noinspection PyPep8Naming
    def get_GT_state(self):
        state = np.insert(self.dmcenv._physics.get_state(), 0, self.dmcenv._step_count)
        return state

    def _get_observation(self):
        return convert_observation(self.timestep.observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.timestep = self.dmcenv.reset()
        return self._get_observation()

    def step(self, a):

        if type(self.action_space) == DmcDiscrete:
            a += self.action_space.offset
        self.timestep = self.dmcenv.step(a)

        return self._get_observation(), self.timestep.reward, self.timestep.last(), {}

    def render(self, close=False):
        self.pixels = self.dmcenv.physics.render(**self.render_mode_list[self.render_mode]["render_kwargs"])
        if close:
            if self.viewer[self.render_mode] is not None:
                self._get_viewer(self.render_mode).close()
                self.viewer[self.render_mode] = None
            return
        elif self.render_mode_list[self.render_mode]["show"]:
            self._get_viewer(self.render_mode).update(self.pixels)

        if self.render_mode_list[self.render_mode]["return_pixel"]:
            return self.pixels

    def _get_viewer(self, mode="human"):
        if self.viewer[mode] is None:
            self.viewer[mode] = DmControlViewer(
                self.pixels.shape[1], self.pixels.shape[0], self.render_mode_list[mode]["render_kwargs"]["depth"]
            )
        return self.viewer[mode]

    def compute_state_difference(self, state1, state2):
        return np.max(state1[1:] - state2[1:])


def create_render_mode(
    name,
    show=True,
    return_pixel=False,
    height=480,
    width=640,
    camera_id=-1,
    overlays=(),
    depth=False,
    scene_option=None,
):
    render_kwargs = {
        "height": height,
        "width": width,
        "camera_id": camera_id,
        "overlays": overlays,
        "depth": depth,
        "scene_option": scene_option,
    }
    render_mode_list[name] = {"show": show, "return_pixel": return_pixel, "render_kwargs": render_kwargs}


render_mode_list = {}
create_render_mode("human", show=True, return_pixel=False)
create_render_mode("rgb_array", show=False, return_pixel=True)
create_render_mode("human_rgb_array", show=True, return_pixel=True)
