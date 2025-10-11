#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reproducible seeds across Python, NumPy, PyTorch, CUDA
- Disables nondeterministic CuDNN ops (optional)
- Provides context manager for temporary seeds
"""

import os
import random
from contextlib import contextmanager

def set_global_seed(seed: int, deterministic_torch: bool = True):
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int):
    """
    For DataLoader workers: torch.utils.data.DataLoader(..., worker_init_fn=seed_worker)
    """
    import numpy as np
    import torch

    base_seed = torch.initial_seed() % 2**32
    random.seed(base_seed)
    np.random.seed(base_seed)

@contextmanager
def temp_seed(seed: int, deterministic_torch: bool = True):
    """
    Temporarily set a seed inside a context, restoring previous state after.
    """
    import numpy as np
    import torch

    # Save state
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()

    # Set
    set_global_seed(seed, deterministic_torch=deterministic_torch)
    try:
        yield
    finally:
        # Restore
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)