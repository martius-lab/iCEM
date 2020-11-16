from importlib import import_module
from .gt_model import AbstractGroundTruthModel, GroundTruthModel
from .gt_par_model import ParallelGroundTruthModel

models_dict = {
        "GroundTruthModel": GroundTruthModel,
        "ParallelGroundTruthModel": ParallelGroundTruthModel
    }


def forward_model_from_string(mod_str: str) -> type:  # Return a class not an instance
    """
    Returns a model class equivalent to the supplied string.
    :param mod_str: Name of model class
    :return: The model class
    """
    if mod_str in models_dict:
        if isinstance(models_dict[mod_str], tuple):
            mod_package, mod_class = models_dict[mod_str]
            module = import_module(mod_package, "models")
            cls = getattr(module, mod_class)
            return cls
        else:
            return models_dict[mod_str]
    else:
        raise NotImplementedError("Implement model class {} and add it to dictionary".format(mod_str))
