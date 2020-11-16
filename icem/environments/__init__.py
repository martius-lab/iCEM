from importlib import import_module


def _check_for_mujoco_lock(env_package):
    if env_package == ".mujoco":
        import cloudpickle
        import os
        import time
        path = os.path.dirname(cloudpickle.__file__)
        site_packages_path = path.split("cloudpickle")[0]
        lock_file = os.path.join(site_packages_path, "mujoco_py", "generated", "mujocopy-buildlock.lock")
        try:
            while os.path.exists(lock_file):
                age_of_lock = time.time() - os.path.getmtime(lock_file)
                if age_of_lock > 300:  # the lock is already 300 seconds old (120 is less for cluster jobs)
                    print(f"Deleting stale mujoco lock in {lock_file}")
                    os.remove(lock_file)
                else:
                    print(f"waiting for mujoco lock to be released (I kill it in {round(300-age_of_lock)}s) {lock_file}")
                    time.sleep(5)
        except:
            pass


def env_from_string(env_string, **env_params):
    env_dict = {
        "DiscreteMountainCar": (".classic", "DiscreteActionMountainCar"),
        "DiscreteCartPole": (".classic", "DiscreteActionCartPole"),
        "ContinuousMountainCar": (".classic", "ContinuousMountainCar"),
        "ContinuousPendulum": (".classic", "ContinuousPendulum"),
        "ContinuousLunarLander": (".classic", "ContinuousLunarLander"),
        # - MUJOCO - #
        "HalfCheetah": (".mujoco", "HalfCheetahMaybeWithPosition"),
        "Reacher": (".mujoco", "Reacher"),
        "Ant": (".mujoco", "Ant"),
        "Hopper": (".mujoco", "Hopper"),
        "HumanoidStandup": (".mujoco", "HumanoidStandup"),
        "Humanoid": (".mujoco", "Humanoid"),
        # - ROBOTICS - #
        "FetchPickAndPlace": (".robotics", "FetchPickAndPlace"),
        "FetchReach": (".robotics", "FetchReach"),
        # - DMSuite - #
        "cartpole": (".dm_suite", "CartPoleSuite"),
        "reacher": (".dm_suite", "ReacherSuite"),
        "restricted_reacher": (".dm_suite", "RestrictedReacherSuite"),
        "point_mass": (".dm_suite", "DoubleIntSuite"),
        "restricted_point_mass": (".dm_suite", "RestrictedDoubleIntSuite"),
        "cheetah": (".dm_suite", "HalfCheetahSuite"),
        "swimmer": (".dm_suite", "SwimmerSuite"),
        # - MJEnvs - #
        "Door": (".mjenvs", "Door"),
        "Relocate": (".mjenvs", "Relocate"),
        # - CausalWorld - #
        "causalworld": (".causalworld", "CausalWorldEnv"),
    }
    if env_string in env_dict:
        env_package, env_class = env_dict[env_string]
        _check_for_mujoco_lock(env_package)
        module = import_module(env_package, "environments")
        cls = getattr(module, env_class)
        env = cls(**env_params, name=env_string)
    else:
        raise ImportError(f"add \'{env_string}\' entry to dictionary")

    return env
