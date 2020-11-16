import os
import pickle
import numpy as np
from abc import ABC
from pathlib import Path

from misc.base_types import Controller, ForwardModel, Pretrainer
from controllers.abstract_controller import TrainableController
from misc.rolloutbuffer import RolloutBuffer


def pretrainer_from_string(trainer_name, trainer_params):
    trainers_dict = {'trajectory': TrajectoryPretrainer
                     }
    if trainer_name not in trainers_dict:
        raise KeyError(f"trainer name '{trainer_name}' not in dictionary entries: {trainers_dict.keys()}")
    return trainers_dict[trainer_name](**trainer_params)


def _parse_no_yes_auto(argument):
    no_yes_auto = 0
    if argument is not None:
        if isinstance(argument, bool) and argument:
            no_yes_auto = 1
        elif isinstance(argument, str):
            if argument == 'yes':
                no_yes_auto = 1
            elif argument == 'auto':
                no_yes_auto = 2
            else:
                raise SyntaxError(f"unknown load argument {argument}, valid: None, True, False, 'yes', 'auto'")
    return no_yes_auto


def file_name_to_absolute_path(file, path, default):
    res = file
    if file is None:
        res = default
    # if the given path is a relative path, use the default path (model_dir)
    if not os.path.isabs(res):
        res = os.path.join(path, res)
    return res


class CheckpointManager:
    def __init__(self, *, model_dir, path="checkpoints", rollouts_file="rollouts",
                 controller_file="controller", forward_model_file="forward_model", reward_dict_file="reward_info.npy",
                 load, save, save_every_n_iter=1, restart_every_n_iter=None,
                 keep_only_last=False, exclude_rollouts=False):
        self.rollouts_file = rollouts_file
        self.base_path = file_name_to_absolute_path(path, path=model_dir, default="checkpoints")
        self.path = self.base_path
        self._check_for_latest()
        self.controller_file = controller_file if controller_file is not None else "controller"
        self.model_file = forward_model_file if forward_model_file is not None else "forward_model"
        self.reward_dict_file = reward_dict_file
        self.save = save
        self.load_no_yes_auto = _parse_no_yes_auto(load)
        self.save_every_n_iter = save_every_n_iter
        self.keep_only_last = keep_only_last
        self.restart_every_n_iter = restart_every_n_iter
        self.do_restarting = self.restart_every_n_iter is not None and self.restart_every_n_iter > 0
        if self.do_restarting:
            assert self.load_no_yes_auto > 0, "load flag needs to be 'auto' or True"
        self.exclude_rollouts = exclude_rollouts
        self.was_controller_loaded = False
        self.was_model_loaded = False
        self.were_buffers_loaded = False
        self.was_reward_dict_loaded = False

    def _check_for_latest(self):
        latest = f"{self.base_path}_latest"
        if os.path.exists(latest):
            self.path = latest

    def update_checkpoint_dir(self, step):
        if self.keep_only_last:
            self.path = self.base_path
        else:
            self.path = f"{self.base_path}_{step:03}"
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def finalized_checkpoint(self):
        # create link to latest checkpoint
        latest = f"{self.base_path}_latest"
        if os.path.islink(latest):
            os.remove(latest)
        if not os.path.exists(latest):
            os.symlink(Path(self.path).name, latest)

    def save_main_state(self, main_state):
        f = os.path.join(self.path, "main_state.npy")
        main_state.save(f)

    def load_main_state(self, main_state):
        f = os.path.join(self.path, "main_state.npy")
        if self.load_no_yes_auto > 0:
            try:
                main_state.load(f)
            except FileNotFoundError as e:
                if self.load_no_yes_auto == 1:
                    raise e
                else:
                    print(f"auto loading: Notice: could not load main state from {f}")

    def store_buffer(self, rollout_buffer: RolloutBuffer, suffix=''):
        if self.rollouts_file is not None and not self.exclude_rollouts:
            with open(os.path.join(self.path, self.rollouts_file + suffix), 'wb') as f:
                pickle.dump(rollout_buffer, f)

    def load_buffer(self, suffix, rollout_buffer: RolloutBuffer):
        if self.rollouts_file is not None and self.load_no_yes_auto > 0 and not self.exclude_rollouts:
            file_path = os.path.join(self.path, self.rollouts_file + suffix)
            try:
                with open(file_path, 'rb') as f:
                    r = pickle.load(f)
                    rollout_buffer.__dict__ = r.__dict__
                    print(f"loaded rollout buffer from {file_path}, buffer size: {len(r)}")
                    self.were_buffers_loaded = True
            except FileNotFoundError as e:
                if self.load_no_yes_auto == 1:  # in 'yes'/True mode it has to load it
                    print(f"Error: could not load rollout buffer from {file_path}")
                    raise e
                else:
                    print(f"auto loading: Notice: could not load rollout buffer from {file_path}")

    def load_controller(self, controller):
        file = os.path.join(self.path, self.controller_file)
        if isinstance(controller, TrainableController):
            if self.load_no_yes_auto == 1:
                controller.load(file)
                self.was_controller_loaded = True
            elif self.load_no_yes_auto == 2:
                try:
                    controller.load(file)
                    self.was_controller_loaded = True
                except FileNotFoundError:
                    print(f"auto loading: Notice: could not load controller from {file}")
        if self.was_controller_loaded:
            print(f"loaded controller from file: {file}")

    def store_controller(self, controller: Controller):
        if self.save and self.controller_file is not None and isinstance(controller, TrainableController):
            controller.save(os.path.join(self.path, self.controller_file))

    def load_forward_model(self, forward_model):
        file = os.path.join(self.path, self.model_file)
        if self.load_no_yes_auto == 1:
            forward_model.load(file)
            self.was_model_loaded = True
        elif self.load_no_yes_auto == 2:
            try:
                forward_model.load(file)
                self.was_model_loaded = True
            except FileNotFoundError:
                print(f"auto loading: Notice: could not load model from {file}")
        if self.was_model_loaded:
            print(f"loaded forward_model from file: {file}")

    def store_forward_model(self, forward_model: ForwardModel):
        if self.save and forward_model and self.model_file is not None:
            forward_model.save(os.path.join(self.path, self.model_file))

    def save_reward_dict(self, reward_dict):
        if self.save and reward_dict and self.reward_dict_file is not None:
            np.save(os.path.join(self.path, self.reward_dict_file), reward_dict)

    def load_reward_dict(self, reward_dict):
        file = os.path.join(self.path, self.reward_dict_file)
        if self.load_no_yes_auto == 1:
            reward_dict = np.load(file).item() if os.path.exists(file) else {}
            self.was_reward_dict_loaded = True
        elif self.load_no_yes_auto == 2:
            try:
                reward_dict = np.load(file).item()
                self.was_reward_dict_loaded = True
            except FileNotFoundError:
                print(f"auto loading: Notice: could not load reward_dict from {file}")
        if self.was_reward_dict_loaded:
            print(f"loaded reward_dict from file: {file}")
        return reward_dict
