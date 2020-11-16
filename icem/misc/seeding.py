import numpy as np
from torch import manual_seed


def create_seed():
    return np.random.randint(0, 2 ** 32)


class Seeding:
    SEED = create_seed()

    @classmethod
    def set_seed(cls, seed=None, *, env):
        cls.SEED = create_seed() if seed is None else seed
        # print(f"Simulated Env created with seed: {cls.SEED}")
        # print(f'Using seed {cls.SEED}')
        env.seed(cls.SEED)
        np.random.seed(cls.SEED)
        manual_seed(cls.SEED)
