from typing import Dict, List, Tuple
import numpy as np
import torch
import pandas as pd

from functools import lru_cache
from pytorch_lightning.core import LightningModule

from utils.data_utils import flatten_var_name_dict, var_names_to_CMIP, add_hpa_zeros_to_var_names

class NetArchitecture(LightningModule):
    """
    Base class for all neural network architectures used in the project.
    """
    def __init__(self,default_in_vars: Dict[str, List[int]], default_out_vars: Dict[str, List[int]], lead_time: str, predict_residuals: bool):
        """
        Args:
            default_in_vars: Dictionary with variable names as keys and levels as values. Levels in hPa
            default_out_vars: Dictionary with variable names as keys and levels as values. Levels in hPa
            lead_time: Time delta string in the format '1H', '1D' to determine the lead time for the model
            predict_residuals: If the models should predict the residuals or the actual values
        """
        super().__init__()


        self.lead_time_hours = int(pd.Timedelta(lead_time)/ np.timedelta64(1, 'h'))
        
        # A bit ugly, variables need to follow same naming scheme as in dataloader.
        # The dataloader however changes variable names from the config file to the CMIP6 standard
        self.in_var_map = {var: i for i, var in enumerate(flatten_var_name_dict(var_names_to_CMIP(add_hpa_zeros_to_var_names(default_in_vars))))}
        self.out_var_map = {var: i for i, var in enumerate(flatten_var_name_dict(var_names_to_CMIP(add_hpa_zeros_to_var_names(default_out_vars))))}

        print('In var map:', self.in_var_map)
        print('Out var map:', self.out_var_map)
        self.predict_residuals = predict_residuals

    def forward(self, x):
        raise NotImplementedError
    
    
    @lru_cache(maxsize=None)
    def get_indices_in_vars(self, var_names: Tuple[str],device: str) -> torch.Tensor:
        "Map each input variable name to a fixed index."
        ids = np.array([self.in_var_map[var] for var in var_names])
        return torch.from_numpy(ids).to(device)
    
    @lru_cache(maxsize=None)
    def get_indices_out_vars(self, var_names: Tuple[str],device: str) -> torch.Tensor:
        "Map each output variable name to a fixed index."
        ids = np.array([self.out_var_map[var] for var in var_names])
        return torch.from_numpy(ids).to(device)

    @lru_cache(maxsize=None)
    def get_indices_prognostics(self, var_names_in: Tuple[str], var_names_prognostics: Tuple[str], device:str) -> torch.Tensor:
        """
        Get the indices of prognostic variables among the input variables.
        """
        var_names_in = list(var_names_in)
        ids = np.array([var_names_in.index(var) for var in var_names_prognostics])
        return torch.from_numpy(ids).to(device)

    
