import os
import pickle

import numpy as np
from collections.abc import Iterable

from misc.rolloutbuffer import RolloutBuffer


def trajectory_reward_fn(reward_fn, states, actions, next_states):
    rewards_path = [
        reward_fn(state, action, next_state) for state, action, next_state in zip(states, actions, next_states)
    ]  # shape: [horizon, num_sim_paths]

    return rewards_path


class ArrayIteratorParallelRowwise(Iterable):
    """creates an iterator for the array that yields elements in a row-wise fashion
     but with the possibility to get several rows in parallel (shape)
     (or all rows in parallel which results in one column at a time) """

    def __init__(self, array: np.ndarray, num_parallel: int):
        self.array = array
        self.num_parallel = num_parallel
        self.subset_time_idx = 0
        if self.num_parallel > self.array.shape[0]:
            raise AttributeError("too many parallel rows requested!")

    def __iter__(self):
        return self

    def __next__(self):
        col_number = self.array.shape[1]  # shape of array: [#max_parallel, horizon, ...]
        if col_number == 0:
            raise AttributeError("I don't have any item(s) left.")
        if self.num_parallel == 1 or self.num_parallel < self.array.shape[0]:
            if self.num_parallel == 1:
                result = self.array[0, self.subset_time_idx]
            else:
                result = self.array[:self.num_parallel, self.subset_time_idx]

            self.subset_time_idx += 1
            if self.subset_time_idx >= col_number:  # we are at the end with the subset. kill the rows from the matrix
                self.subset_time_idx = 0
                self.array = self.array[self.num_parallel:]
        else:
            assert(self.num_parallel == self.array.shape[0])  # fully parallel case
            result = self.array[:, 0]
            self.array = self.array[:, 1:]
        return result


class ParallelRowwiseIterator:

    def __init__(self, sequences: np.ndarray):
        """:param sequences: shape: [p, h, d]
        """
        self.sequences = sequences
        self.sequence_iterator = None

    @staticmethod
    def get_num_parallel(obs):
        if obs.ndim == 1:
            return 1
        else:
            return obs.shape[0]

    def get_next(self, obs: np.ndarray):
        """ Every time get_action is called we take the actions from the actions_sequence and return it.
        In case we are asked to return fewer (parallel) actions then we are set up for (p above)
        then we first continue this amount of roll-outs and then proceed to the next sub-batch
        :param obs: shape [p, d] (number parallel runs, state-dim)
        """
        if self.sequence_iterator is None:
            self.sequence_iterator = ArrayIteratorParallelRowwise(self.sequences, self.get_num_parallel(obs))
        return self.sequence_iterator.__next__()
