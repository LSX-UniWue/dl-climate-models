import torch

from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet as SFNO
from models.base_model import NetArchitecture

class SFNO2DModule(NetArchitecture):
    """
    A Spherical Fourier Neural Operator implementation from the torch harmonics package:
    https://github.com/NVIDIA/torch-harmonics/tree/main    
    """

    def __init__(
        self,
        default_in_vars,
        default_out_vars,
        lead_time,
        predict_residuals,
        spectral_transform='sht',
        grid="legendre-gauss",
        num_layers=4,
        scale_factor=3,
        embed_dim=256,
        operator_type='driscoll-healy',
        height: int = 32,
        width: int = 64,
        hard_thresholding_fraction: float = 1.0,
        factorization: str = None,
        rank: float = 1.0,
        big_skip: bool = False,
        pos_embed: bool = False,
        use_mlp: bool = False,
        normalization_layer: str = None,
        drop_rate:float = 0.0
    ):
        super(SFNO2DModule, self).__init__(default_in_vars, default_out_vars, lead_time, predict_residuals)

        in_channels = len(self.in_var_map)

        print("SFNO Number in Channels: ", in_channels)
        
        self.sfno = SFNO(
            in_chans=in_channels,
            out_chans=len(self.out_var_map),
            spectral_transform=spectral_transform,
            img_size=(height, width),
            grid=grid,
            num_layers=num_layers,
            scale_factor=scale_factor,
            embed_dim=embed_dim,
            operator_type=operator_type,
            hard_thresholding_fraction=hard_thresholding_fraction,
            factorization=factorization,
            rank=rank,
            big_skip=big_skip,
            pos_embed=pos_embed,
            use_mlp=use_mlp,
            normalization_layer=normalization_layer,
            drop_rate=drop_rate,
        )

    def forward(self, x_in,var_names_in, var_names_out, var_names_prognostics
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: `[B, V_in, H, W]` shape. Input weather/climate variables
            var_names_in: List of variable names in x along channel dimension.
            var_names_out: List of variable names in the output. Necessary when there are diagnostic variables, but not yet implemented
            var_names_prognostics: List of variable names in the prognostic variables.

        Returns:
            preds (torch.Tensor): `[B, V_out, H, W]` shape. Predicted weather/climate variables.
        """
        # Forward input through model
        x = self.sfno(x_in)
        
        if self.predict_residuals:
            # Get the indices of prognostic variables among the input variables
            prognostic_var_ids = self.get_indices_prognostics(var_names_in, var_names_prognostics, x.device)
            x= x_in[:,prognostic_var_ids] + x

        return x
