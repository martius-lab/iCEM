from importlib import import_module

from . import abstract_controller


def controller_from_string(controller_str):
    return ControllerFactory(controller_str=controller_str)


class ControllerFactory:
    # noinspection SpellCheckingInspection
    valid_base_controllers = {
        "mpc-icem": (".icem", "MpcICem"),
        "mpc-cem-std": (".mpc", "MpcCemStd"),
        "mpc-random": (".mpc", "MpcRandom"),
        "random": (".random", "RndController")
    }

    controller = None

    def __new__(cls, *, controller_str):

        if controller_str in cls.valid_base_controllers:
            ctrl_package, ctrl_class = cls.valid_base_controllers[controller_str]
            module = import_module(ctrl_package, "controllers")
            cls.controller = getattr(module, ctrl_class)
        else:
            raise ImportError(f"cannot find \'{controller_str}\' in known controller: "
                              f"{cls.valid_base_controllers.keys()}")

        return cls.controller
