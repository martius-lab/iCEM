import json
import os
import re
import sys
from collections import Mapping
from copy import deepcopy
from warnings import warn
import ast

import numpy as np
import inspect
from contextlib import contextmanager
import tqdm

from abc import ABC, abstractmethod

from functools import update_wrapper, partial


class Decorator(ABC):
    def __init__(self, f):
        self.func = f
        update_wrapper(self, f, updated=[])  # updated=[] so that 'self' attributes are not overwritten

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __get__(self, instance, owner):
        new_f = partial(self.__call__, instance)
        update_wrapper(new_f, self.func)
        return new_f


def sin_and_cos_to_radians(sin_of_angle, cos_of_angle):
    theta = np.arccos(cos_of_angle)
    theta *= np.sign(sin_of_angle)
    return theta


@contextmanager
def redirect_stdout__to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        to_print = "".join(map(repr, args))
        tqdm.tqdm.write(to_print, **kwargs)

    try:
        # Globally replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


def flatten_list_one_level(list_of_lists):
    return [x for lst in list_of_lists for x in lst]


def tqdm_context(*args, **kwargs):
    with redirect_stdout__to_tqdm():
        postfix_dict = kwargs.pop('postfix_dict', {})
        additional_info_flag = kwargs.pop('additional_info_flag', False)

        t_main = tqdm.tqdm(*args, **kwargs)
        t_main.postfix_dict = postfix_dict
        if additional_info_flag:
            yield t_main
        for x in t_main:
            t_main.set_postfix(**t_main.postfix_dict)
            t_main.refresh()
            yield x


def delegates(to=None, keep=False):
    """Decorator: replace `**kwargs` in signature with params from `to`"""

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        kwargs = sigd['kwargs']
        del(sigd['kwargs'])
        s2 = {k: v for k, v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep:
            sigd['kwargs'] = kwargs
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def recursive_objectify(nested_dict, make_immutable=True):
    """Turns a nested_dict into a nested ParamDict"""
    result = deepcopy(nested_dict)
    for k, v in result.items():
        if isinstance(v, Mapping):
            result = dict(result)
            result[k] = recursive_objectify(v, make_immutable)
    if make_immutable:
        returned_result = ParamDict(result)
    else:
        returned_result = dict(result)
    return returned_result


def update_recursive(d, u, defensive=False):
    for k, v in u.items():
        if defensive and k not in d:
            raise KeyError("Updating a non-existing key")
        if isinstance(v, Mapping):
            d[k] = update_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def is_json_file(cmd_line):
    try:
        return os.path.isfile(cmd_line)
    except Exception as e:
        warn('JSON parsing suppressed exception: ', e)
        return False


def is_parseable_dict(cmd_line):
    try:
        res = ast.literal_eval(cmd_line)
        return isinstance(res, dict)
    except Exception as e:
        warn('Dict literal eval suppressed exception: ', e)
        return False


def resolve_params_hierarchy(init_params, verbose=True):
    def to_list(inp, settings_filepath):
        if not isinstance(inp, (list, tuple)):
            inp = [inp]

        if len(inp) == 0:
            return []

        return [os.path.join(settings_file_path, re.sub(r'(\w)\.(\w)', r'\1/\2', k.replace('..', '../')) + '.json') for
                k in
                inp if k is not None]

    params_hierarchy = []
    settings_files = []
    inherits_from_query = []

    if is_json_file(sys.argv[1]):
        print(1)
        settings_file_path = os.path.dirname(os.path.abspath(sys.argv[1]))
        settings_files.append(os.path.abspath(sys.argv[1]))
    elif is_parseable_dict(sys.argv[1]):
        print(2)
        settings_file_path = os.path.dirname(init_params.default_json)
        settings_files.append(os.path.abspath(init_params.default_json))
    else:
        raise ValueError('Failed to parse command line')
    inherits_from_query.extend(to_list(init_params.get('inherits_from', []), settings_file_path))
    while inherits_from_query:
        inherited_settings_file = inherits_from_query.pop()
        if inherited_settings_file in settings_files:
            continue
        settings_files.append(inherited_settings_file)
        with open(inherited_settings_file, 'r') as f:
            params_hierarchy.append(json.load(f))
        if params_hierarchy[-1].get('inherits_from', None) is None:
            continue
        settings_file_path = os.path.dirname(os.path.abspath(inherited_settings_file))
        inherits_from_query.extend(to_list(params_hierarchy[-1].get('inherits_from', []), settings_file_path))
    params_hierarchy.append(init_params)

    params = {}
    for p in params_hierarchy:
        update_recursive(params, p)

    params = recursive_objectify(params)

    if verbose:
        print(params)

    return params


def compute_and_log_reward_info(rollouts, logger, prefix="", exec_time=None):
    reward_info = {
        prefix + "mean_avg_reward": rollouts.mean_avg_reward,
        prefix + "mean_max_reward": rollouts.mean_max_reward,
        prefix + "mean_return": rollouts.mean_return,
        prefix + "std_return": rollouts.std_return,
    }
    if exec_time is not None:
        reward_info.update({prefix+"exec_time": exec_time})
    try:
        reward_info[prefix + 'mean_success'] = np.mean(rollouts.as_array('successes')[:, -1])
        reward_info[prefix + 'std_success'] = np.std(rollouts.as_array('successes')[:, -1])
    except TypeError:
        pass
    for k, v in reward_info.items():
        logger.log(v, key=k, to_tensorboard=True)
        logger.info(f'{k}: {v}', to_stdout=True)

    return reward_info


def update_reward_dict(iteration, reward_info: dict, reward_dict: dict):
    if 'step' in reward_dict:
        reward_dict['step'].append(iteration)
    else:
        reward_dict.update({'step': [iteration]})
    for item in reward_info:
        if item in reward_dict:
            reward_dict[item].append(reward_info[item])
        else:
            reward_dict.update({item: [reward_info[item]]})
    return reward_dict


class ParamDict(dict):
    """ An immutable dict where elements can be accessed with a dot"""

    def __getattr__(self, *args, **kwargs):
        try:
            return self.__getitem__(*args, **kwargs)
        except KeyError as e:
            raise AttributeError(e)

    def __delattr__(self, item):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __setattr__(self, key, value):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __setitem__(self, key, value):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __deepcopy__(self, memo):
        """ In order to support deepcopy"""
        return ParamDict([(deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items()])

    def __repr__(self):
        return json.dumps(self, indent=4, sort_keys=True)

    def get_pickleable(self):
        return recursive_objectify(self, make_immutable=False)

