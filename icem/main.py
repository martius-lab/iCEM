import os
from collections import deque
import numpy as np
import json
import time
import allogger
from smart_settings.param_classes import recursive_objectify
from controllers import controller_from_string
from controllers.abstract_controller import ModelBasedController, NeedsPolicyController, TeacherController, \
    ParallelController
from environments import env_from_string
from misc.helpers import tqdm_context, resolve_params_hierarchy, compute_and_log_reward_info, \
    update_reward_dict, update_from_cmd_line, save_settings_to_json
from misc.initialization import CheckpointManager
from models import forward_model_from_string
from misc.rollout_utils import RolloutManager
from misc.rolloutbuffer import RolloutBuffer
from misc.seeding import Seeding
import torch.multiprocessing

valid_data_sources = {"env", "policy", "expert"}
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ.update({"OMP_NUM_THREADS": "1"})


def get_controllers(params, env, forward_model):
    if params.initial_controller is None or params.initial_controller in ["none", 'null', None]:
        initial_controller = None
    else:
        controller_class = controller_from_string(params.initial_controller)
        if issubclass(controller_class, ModelBasedController):
            initial_controller = controller_class(
                env=env, forward_model=forward_model, **params.initial_controller_params
            )
        else:
            initial_controller = controller_class(env=env, **params.initial_controller_params)

    controller_class = controller_from_string(params.controller)
    if issubclass(controller_class, ParallelController):
        controller_params = recursive_objectify(params.controller_params, make_immutable=False)
    else:
        controller_params = params.controller_params
    if issubclass(controller_class, ModelBasedController):
        main_controller = controller_class(env=env, forward_model=forward_model, **controller_params)
    else:
        main_controller = controller_class(env=env, **controller_params)
    if main_controller.needs_data:
        if params.controller_data_sources is None or len(params.controller_data_sources) < 1:
            raise AttributeError("controller needs data to be trained but no source given")
        for s in params.controller_data_sources:
            if s not in valid_data_sources:
                raise KeyError(f"Invalid data source \'{s}\', valid ones are {''.join(valid_data_sources)}")

    return initial_controller, main_controller


class MainState:
    def __init__(self, iteration, successful_rollouts):
        self.iteration = iteration
        self.successful_rollouts = successful_rollouts

    def save(self, file):
        np.save(file, (self.iteration, self.successful_rollouts, dict(allogger.get_logger('root').step_per_key)))
        print(f"checkpointing at iteration {self.iteration}")

    def load(self, file):
        dat = np.load(file, allow_pickle=True)
        if len(dat) == 1:
            (self.iteration,) = dat
        elif len(dat) == 2:
            (self.iteration, self.successful_rollouts) = dat
        elif len(dat) == 3:
            (self.iteration, self.successful_rollouts, step_per_key) = dat
            allogger.get_logger('root').step_per_key = allogger.get_logger('root').manager.dict(step_per_key)
        else:
            raise AttributeError(
                f"loading if main_state failed from {file} got collection of len {len(dat)} expect 1 or 2 or 3!")
        self.iteration += 1  # we want to start with the next iteration
        print(f"loaded checkpoint and starting at iteration {self.iteration}")


def main():
    params = update_from_cmd_line()
    params = resolve_params_hierarchy(params)
    allogger.basic_configure(logdir=params.model_dir, default_outputs=['tensorboard'])
    allogger.utils.report_env(to_stdout=True)

    save_settings_to_json(params, params.model_dir)

    reward_info = {}
    reward_info_full = {}
    min_horizon = []
    average_return_history = deque(maxlen=10)
    min_time_required_to_solve = params.training_iterations
    env = env_from_string(params.env, **params.env_params)
    logger = allogger.get_logger(scope="main")
    main_state = MainState(0, 0)
    potentially_restart = False

    Seeding.set_seed(params.seed if "seed" in params else None, env=env)

    allogger.get_logger('root').info(f'Using seed {Seeding.SEED}')
    allogger.get_logger('root').info(json.dumps(params, indent=2))

    forward_model = (
        None
        if params.forward_model == "none"
        else forward_model_from_string(params.forward_model)(env=env, **params.forward_model_params)
    )

    initial_controller, main_controller = get_controllers(params, env, forward_model)
    os.makedirs(params.model_dir, exist_ok=True)

    rollout_buffer = RolloutBuffer()  # buffer for main controller/policy rollouts
    rollout_buffer_expert = RolloutBuffer()  # buffer for successful expert rollouts and dagger relabels
    rollout_buffer_expert_all = RolloutBuffer()  # buffer for all expert rollouts
    rollout_buffer_eval = RolloutBuffer()  # buffer for policy evaluation rollouts

    if "checkpoints" in params:  # we could check whether we want to check for rollout_length consistency?
        checkpoint_manager = CheckpointManager(model_dir=params.model_dir, **params.checkpoints)
        checkpoint_manager.load_buffer(suffix='', rollout_buffer=rollout_buffer)
        if "evaluation_rollouts" in params and params.evaluation_rollouts > 0:
            checkpoint_manager.load_buffer(suffix='_eval', rollout_buffer=rollout_buffer_eval)

        if forward_model:
            checkpoint_manager.load_forward_model(forward_model)
        checkpoint_manager.load_controller(main_controller)
        reward_info_full = checkpoint_manager.load_reward_dict(reward_info_full)
        checkpoint_manager.load_main_state(main_state)
    else:
        checkpoint_manager = CheckpointManager(model_dir=params.model_dir, load=False, save=False)

    need_pretrained_checkpoint = False

    # function that we use for saving a checkpoint
    def save_checkpoint(cpm: CheckpointManager, final=False):
        step = main_state.iteration
        if cpm is not None and cpm.save:
            if final or step % cpm.save_every_n_iter == 0:
                cpm.update_checkpoint_dir(step)
                cpm.save_main_state(main_state)
                for _rollouts, _name in [(rollout_buffer, ''), (rollout_buffer_eval, '_eval'),
                                         (rollout_buffer_expert, '_expert'),
                                         (rollout_buffer_expert_all, '_expert_all')]:
                    if len(_rollouts) > 0:
                        cpm.store_buffer(rollout_buffer=_rollouts, suffix=_name)

                cpm.store_forward_model(forward_model)
                cpm.store_controller(main_controller)
                cpm.save_reward_dict(reward_info_full)
                cpm.finalized_checkpoint()

    if need_pretrained_checkpoint:
        main_state.iteration -= 1
        save_checkpoint(checkpoint_manager)
        main_state.iteration += 1

    postfix_dict = {"Successful Rollouts": main_state.successful_rollouts} if main_state.successful_rollouts else {}
    do_initial_rollouts = initial_controller is not None and params.initial_number_of_rollouts > 0

    total_iterations = params.training_iterations + 1 * do_initial_rollouts
    if checkpoint_manager.were_buffers_loaded:
        do_initial_rollouts = False
    current_max_iterations = total_iterations
    if checkpoint_manager.do_restarting:
        if main_state.iteration + checkpoint_manager.restart_every_n_iter < total_iterations:
            current_max_iterations = main_state.iteration + checkpoint_manager.restart_every_n_iter \
                                     + 1 * do_initial_rollouts
            print(f"Due to restarting we are only running {checkpoint_manager.restart_every_n_iter} iterations now")
            potentially_restart = True

    # Rollout Manager initialization
    rollout_man = RolloutManager(env, params.rollout_params)
    t_main = (tqdm_context(range(main_state.iteration, current_max_iterations),
                           initial=main_state.iteration, total=total_iterations, desc="training_it",
                           postfix_dict=postfix_dict, additional_info_flag=True))
    gen_main = next(t_main)

    for iteration in t_main:  # first iteration is for initial controller...
        allogger.get_logger('root').info(f'Current iteration: {iteration}')
        main_state.iteration = iteration
        is_init_iteration = (do_initial_rollouts and iteration == 0)
        start_time = time.time()

        if iteration == 0 and do_initial_rollouts:
            controller = initial_controller
            number_of_rollouts = params.initial_number_of_rollouts
            render = params.rollout_params.render_initial
        else:
            controller = main_controller
            number_of_rollouts = params.number_of_rollouts
            render = params.rollout_params.render

        # Execute rollouts
        new_rollouts = RolloutBuffer(
            rollouts=rollout_man.sample(controller, render=render, mode="train", name="train",
                                        no_rollouts=number_of_rollouts)
        )
        reward_info.update(compute_and_log_reward_info(
            new_rollouts, logger, prefix="train_", exec_time=time.time()-start_time))

        # Data processing and gathering
        if params.append_data:
            rollout_buffer.extend(new_rollouts)
        else:
            rollout_buffer = new_rollouts

        # Train
        if forward_model is not None:
            forward_model.train(rollout_buffer)

        # Evaluation runs
        if not is_init_iteration and "evaluation_rollouts" in params and params.evaluation_rollouts > 0:
            new_rollouts = RolloutBuffer(
                rollouts=
                rollout_man.sample(controller, render=params.rollout_params.render_eval,
                                   mode="evaluate", name="eval", no_rollouts=params.evaluation_rollouts,
                                   desc="eval_rollout")
            )
            if "append_data_eval" in params and params.append_data_eval:
                rollout_buffer_eval.extend(new_rollouts)
            else:
                rollout_buffer_eval = new_rollouts
            reward_info.update(compute_and_log_reward_info(new_rollouts, logger, prefix="eval_"))

        if "avg_return_required_to_solve" in params:
            average_return_history.append(reward_info["mean_return"])
            if all([avg_return >= params.avg_return_required_to_solve for avg_return in average_return_history]):
                min_time_required_to_solve = min(min_time_required_to_solve, main_state.iteration)
            reward_info["required_iterations_to_solve"] = min_time_required_to_solve
            logger.info(f'Required iterations to solve: {reward_info["required_iterations_to_solve"]}')

        reward_info_full.update(update_reward_dict(iteration, reward_info, reward_info_full))
        save_checkpoint(checkpoint_manager)
        gen_main.postfix_dict = {
            "Successful Rollouts": main_state.successful_rollouts} if main_state.successful_rollouts else {}

    env.close()
    save_checkpoint(checkpoint_manager, final=True)

    print(reward_info_full)

    allogger.close()


if __name__ == "__main__":
    exit(main())
