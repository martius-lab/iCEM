import os
import sys
from warnings import warn
import numpy as np
from itertools import chain

import imageio
from PIL import Image
from tqdm import tqdm

import allogger

from misc.base_types import Controller, Env
from environments.abstract_environments import GroundTruthSupportEnv, GoalSpaceEnvironmentInterface
from misc.helpers import tqdm_context
from misc.rolloutbuffer import Rollout
# noinspection PyUnresolvedReferences
from misc.rolloutbuffer import RolloutBuffer, _CustomList  # just for pickling
from misc.seeding import Seeding
from misc.parallel_utils import CloudPickleWrapper, clear_mpi_env_vars
from models import GroundTruthModel
from controllers.abstract_controller import ParallelController


class RolloutManager:
    dir_name: str

    valid_modes = ["train", "evaluate"]

    def __init__(self, env, roll_params):
        self.env = env
        self.task_horizon = roll_params.task_horizon
        self.record = roll_params.record
        self.only_final_reward = False if "only_final_reward" not in roll_params else roll_params.only_final_reward
        self.use_env_states = roll_params.use_env_states
        self.video = None
        self.video_path = None
        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        self.num_parallel = roll_params.num_parallel if "num_parallel" in roll_params else 1
        self.parallel_training = roll_params.parallel_training if "parallel_training" in roll_params else False

        self.calls_counter = 0

        self.init_physics_state = None

        if self.num_parallel > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self.num_parallel)])
            self.ps = [
                ctx.Process(target=RolloutManager.worker,
                            args=(Seeding.SEED, i, work_remote, remote, CloudPickleWrapper(self.env)))
                for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes))
            ]
            for p in self.ps:
                p.daemon = True  # if the main process crashes, we should not cause things to hang
                with clear_mpi_env_vars():
                    p.start()

    def __del__(self):
        if self.num_parallel > 1:
            for remote in self.remotes:
                remote.send(("_close", ()))
            for p in self.ps:
                p.join()

    def reset(self):
        self.calls_counter = 0

    def setup_video(self, name_suffix=''):
        self.dir_name = self.logger.logdir
        os.makedirs(self.dir_name, exist_ok=True)
        file_path = os.path.join(self.dir_name, f"rollout_{name_suffix}_{0}_{self.calls_counter:02d}.mp4")
        i = 0
        while os.path.isfile(file_path):
            i += 1
            file_path = os.path.join(self.dir_name, f"rollout_{name_suffix}_{i}_{self.calls_counter:02d}.mp4")
        print("Record video in {}".format(file_path))
        # noinspection SpellCheckingInspection
        return (imageio.get_writer(file_path, fps=self.env.get_fps(),
                                   codec="mjpeg", quality=10, pixelformat="yuvj444p"), file_path)

    # parallel rollouts if we are configured to run in parallel:
    #   evaluation is then always parallel, training only if enabled
    def do_run_in_parallel(self, policy, mode):
        return self.num_parallel > 1 and isinstance(policy, ParallelController) and \
               (self.parallel_training or mode == "evaluate")

    def sample(self, policy: Controller, render: bool, mode="train", name='',
               start_ob=None, start_state=None, no_rollouts=1, use_tqdm=True, desc="rollout_num"):
        if self.env.supports_live_rendering and render and self.record:
            self.video, self.video_path = self.setup_video(name)
            self.env.prepare_for_recording()
            self.calls_counter += 1
        elif self.record and not render:
            print("Cannot record video with render set to false.")
        else:
            self.video = None
        if hasattr(self.env, "dmcenv"):
            if self.do_run_in_parallel(policy, mode):
                warn("Parallel rollouts not implemented for deepmind suite")
            temp_start_ob = [None] * no_rollouts if start_ob is None else start_ob
            temp_start_state = [None] * no_rollouts if start_state is None else start_state
            return [self.sample_deepmindsuite(policy, self.logger, render, mode, temp_start_ob[i], temp_start_state[i])
                    for i in (tqdm_context(range(no_rollouts), desc=desc) if use_tqdm else range(no_rollouts))]
        else:
            if self.do_run_in_parallel(policy, mode):
                assert isinstance(policy, ParallelController)
                return self.par_sample(policy, render, mode, start_ob, start_state, no_rollouts)
            else:
                temp_start_ob = [None] * no_rollouts if start_ob is None else start_ob
                temp_start_state = [None] * no_rollouts if start_state is None else start_state
                return [self.sample_env(policy, self.logger, render, mode, temp_start_ob[i], temp_start_state[i])
                        for i in (tqdm_context(range(no_rollouts), desc=desc) if use_tqdm else range(no_rollouts))]

    def create_sample_params_dict(self, use_tqdm=True):
        return {"use_env_states": self.use_env_states,
                "task_horizon": self.task_horizon, "use_tqdm": use_tqdm,
                "only_final_reward": self.only_final_reward,
                "video": self.video, "video_path": self.video_path
                }

    def sample_env(self, policy, logger, render: bool, mode, start_ob, start_state):
        # this method is also used in the parallel threads that is why it is purely functional (no state)
        return RolloutManager._sample(env=self.env, policy=policy, logger=logger, render=render, mode=mode,
                                      start_ob=start_ob, start_state=start_state,
                                      **self.create_sample_params_dict(True))

    def par_sample(self, policy, render: bool, mode, start_ob, start_state, no_rollouts=1):

        chunks = np.array_split(range(no_rollouts), self.num_parallel)
        chunks = [c for c in chunks if len(c) > 0]
        policies = [policy.get_parallel_policy_copy(c) for c in chunks]
        temp_start_ob = [None] * no_rollouts if start_ob is None else start_ob
        temp_start_state = [None] * no_rollouts if start_state is None else start_state
        start_obs_chunks = [[temp_start_ob[i] for i in c] for c in chunks]
        start_state_chunks = [[temp_start_state[i] for i in c] for c in chunks]

        asked_remotes = []
        for remote, sub_policy, sub_start_obs, sub_start_states \
                in zip(self.remotes, policies, start_obs_chunks, start_state_chunks):
            args = {"policy": sub_policy, "render": render, "mode": mode, "start_obs": sub_start_obs,
                    "start_states": sub_start_states, "logger": None}
            args.update(self.create_sample_params_dict(False))
            remote.send(("_sample", args))
            asked_remotes.append(remote)
        rollout_list = [remote.recv() for remote in asked_remotes]
        if "MujocoException" in rollout_list:
            from mujoco_py import MujocoException
            raise MujocoException
        all_rollouts = list(chain.from_iterable([rollout for rollout in rollout_list]))
        return all_rollouts

    @staticmethod
    def _sample(*, env, policy, logger, render: bool, mode, start_ob, start_state,
                use_env_states, task_horizon, use_tqdm, only_final_reward, video=None, video_path=None):
        if start_ob is not None and isinstance(env, GroundTruthSupportEnv):
            if start_state is None:
                env.set_state_from_observation(start_ob)
            else:
                env.set_GT_state(start_state)
            ob = start_ob
        else:
            ob = env.reset_with_mode(mode)

        if policy.has_state:
            policy.beginning_of_rollout(observation=ob, state=RolloutManager.supply_env_state(env, use_env_states),
                                        mode=mode)
        transitions = []
        steps = 0
        video_frame_file = None
        _return = 0.0
        for t in (tqdm(range(task_horizon), desc="time_steps") if use_tqdm else range(task_horizon)):
            if render:
                if video is not None:
                    frame = env.render(mode="rgb_array")
                    video.append_data(frame)

                    # Workaround to render when recording. Just open video_name.png
                    img = Image.fromarray(frame, mode="RGB")
                    video_frame_file = f"{os.path.splitext(video_path)[0]}.png"
                    img.save(video_frame_file)
                else:
                    env.render()
            state = RolloutManager.supply_env_state(env, use_env_states)
            try:
                ac = policy.get_action(ob, state=state, mode=mode)
                next_ob, rew, done, _ = env.step(ac)
            except Exception as e:
                if e.__class__.__name__ == "MujocoException":
                    warn(f"Got MujocoException {e}. Skipping to next rollout.")
                    break
                else:
                    raise e
            if only_final_reward and t < (task_horizon - 1):
                rew = 0

            transition = [ob, next_ob, ac, rew, done, state]
            if isinstance(env, GoalSpaceEnvironmentInterface):
                transition.append(env.is_success(ob[None, :], ac[None, :], next_ob[None, :])[0])

            transitions.append(tuple(transition))

            ob = next_ob

            steps += 1
            _return += rew
            if logger is not None:
                logger.log(rew, key="reward")
                logger.log(_return, key="return")

            if done:
                break

        if policy.has_state:
            policy.end_of_rollout(total_time=steps, total_return=_return, mode=mode)

        fields = ["observations", "next_observations", "actions", "rewards", "dones", "env_states"]
        if isinstance(env, GoalSpaceEnvironmentInterface):
            fields.append("successes")

        rollout = Rollout(field_names=tuple(fields), transitions=transitions)

        if video is not None and render:
            video.close()
            os.remove(video_frame_file)
        return rollout

    # we don't have parallel execution for DM envs yet
    def sample_deepmindsuite(self, policy, logger, render: bool, mode, start_ob, start_state):
        from environments.dm2gym import DmControlWrapper
        assert isinstance(self.env, DmControlWrapper)
        self.env.dmcenv._step_limit = float("inf")
        if start_ob is not None and isinstance(self.env, GroundTruthSupportEnv):
            if start_state is None:
                self.env.set_state_from_observation(start_ob)
            else:
                self.env.set_GT_state(start_state)
            ob = start_ob
        else:
            ob = self.env.reset_with_mode(mode)

        if policy.has_state:
            policy.beginning_of_rollout(observation=ob,
                                        state=RolloutManager.supply_env_state(self.env, self.use_env_states),
                                        mode=mode)
        self.init_physics_state = self.env.dmcenv.physics.get_state()
        transitions = []
        steps = 0
        _return = 0.0
        for t in tqdm(range(self.task_horizon), desc="time_steps"):

            state = RolloutManager.supply_env_state(self.env, self.use_env_states)
            ac = policy.get_action(ob, state=state, mode=mode)

            next_ob, rew, done, _ = self.env.step(ac)

            if self.only_final_reward and t < (self.task_horizon - 1):
                rew = 0

            transition = [ob, next_ob, ac, rew, done, state]
            if isinstance(self.env, GoalSpaceEnvironmentInterface):
                transition.append(self.env.is_success(ob[None, :], ac[None, :], next_ob[None, :])[0])

            transitions.append(tuple(transition))

            ob = next_ob

            steps += 1
            _return += rew
            if logger is not None:
                logger.log(rew, key="reward")
                logger.log(_return, key="return")

            if done:
                break
        if policy.has_state:
            policy.end_of_rollout(total_time=steps, total_return=_return, mode=mode)
        fields = ["observations", "next_observations", "actions", "rewards", "dones", "env_states"]
        if isinstance(self.env, GoalSpaceEnvironmentInterface):
            fields.append("successes")

        rollout = Rollout(field_names=tuple(fields), transitions=transitions)

        if render:
            from dm_control import viewer

            def gen():
                yield from rollout["actions"]

            it = gen()

            def replay_actions(time_step):
                del time_step
                return next(it)

            self.env.dmcenv._step_count = 0
            self.env.dmcenv._step_limit = len(rollout["actions"])
            self.env.dmcenv.physics.set_state(self.init_physics_state)  # with _?
            self.env.dmcenv.physics.after_reset()
            viewer.launch(environment_loader=self.env.dmcenv, policy=replay_actions)

        return rollout

    @staticmethod
    def supply_env_state(env, use_env_states):
        if use_env_states and isinstance(env, GroundTruthSupportEnv):
            return env.get_GT_state()
        else:
            return None

    @staticmethod
    def worker(seed, worker_id, remote, parent_remote, env_wrapper):
        parent_remote.close()
        orig_env = env_wrapper.x
        from environments import env_from_string
        env = env_from_string(orig_env.name, **orig_env.init_kwargs)
        Seeding.set_seed(seed + worker_id, env=env)

        try:
            while True:
                cmd, data = remote.recv()
                if cmd == "_sample":
                    # only rendering/recording first evaluation rollout
                    if worker_id > 0:
                        data["render"] = False
                    data["env"] = env

                    rollouts = []
                    args = data.copy()
                    _, _ = args.pop("start_obs", None), args.pop("start_states", None)
                    for start_ob, start_state in zip(data["start_obs"], data["start_obs"]):
                        args["start_ob"], args["start_state"] = start_ob, start_state
                        rollout = RolloutManager._sample(**args)
                        rollouts.append(rollout)
                    remote.send(rollouts)
                elif cmd == "_close":
                    break
                else:
                    raise NotImplementedError("cmd: {}".format(cmd))
        except KeyboardInterrupt:
            print("Parallel rollout workers: got KeyboardInterrupt")
        finally:
            env.close()
            remote.close()


class ImitationLearning:
    def __init__(self, tqdm_cntxt, **params):
        self.tqdm_context = tqdm_cntxt
        self._parse_params(**params)

    def _parse_params(self, *, expert_controller, expert_params, do_rollouts, dagger,
                      use_policy_guidance_for_supervision,
                      dagger_params=None):
        self.expert_controller_name = expert_controller
        self.expert_params = expert_params
        self.use_policy_guidance_for_supervision = use_policy_guidance_for_supervision

        self.dagger = dagger_from_string(dagger, dagger_params)

        self.do_rollouts = do_rollouts
        assert self.do_rollouts or dagger != "Smart", "Expert rollouts are needed for SmartDagger"

    def relabel_rollouts(self, *, expert_controller, policy_rollouts, expert_rollouts, policy, forward_model,
                         training_data):
        return self.dagger.relabel_rollouts(expert_controller=expert_controller, policy_rollouts=policy_rollouts,
                                            expert_rollouts=expert_rollouts, policy=policy, training_data=training_data,
                                            forward_model=forward_model, tqdm_context=self.tqdm_context)
