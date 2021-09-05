import numpy as np
import torch


def _state_to_torch(state: np.array):
    """Convert a state vector returned by unity env to a torch tensor"""

    state = torch.from_numpy(state).float().unsqueeze(0)

    return state
