"""
Metrics module for training and evaluation of the model.

"""

import numpy as np
import torch



class MetricsModule(torch.nn.Module):
    def __init__(
            self,
            train_metrics: dict,
            eval_metrics: list,
            rollout_metrics: list = [],
            variable_weights: dict = {},
            masking_values: dict = {},
            climatology_path = None,
            area_weighted = True,

    ):
        super().__init__()
        
        self.area_weighted = area_weighted

        if 'acc' in train_metrics or 'acc' in eval_metrics:
            assert climatology_path is not None, "Climatology path must be provided for ACC"
            raise NotImplementedError("ACC not implemented")
        

        self.train_metrics = []
        sum_weights = sum([w for w in train_metrics.values()])
        for metric_name,weight in train_metrics.items():
            self.train_metrics.append((
                metric_name, eval(f"self.{metric_name}"), weight/sum_weights))
            # name is necessary for accessing the summed metric over all variables from output dict
        
        self.eval_metrics = []
        for metric_name in eval_metrics:
            self.eval_metrics.append(eval(f"self.{metric_name}"))

        self.rollout_metrics = []
        for metric_name in rollout_metrics:
            self.rollout_metrics.append(eval(f"self.{metric_name}"))
        self.masking_values_dict = masking_values
        self.variable_weights = variable_weights

    def set_lat(self, lat, device = 'cpu'):
        if self.area_weighted:
            assert lat is not None, "Latitude array must be provided for area weighted metrics"
            w_lat = np.cos(np.deg2rad(lat))
            w_lat = w_lat / w_lat.mean()  # (H, )
            self.area_weights = torch.from_numpy(w_lat[None,None,:,None]).to(device).float()
        else:
            self.area_weights = 1

    def mse(self, pred, y, var_names):
        """Latitude weighted mean squared error

        Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.
        # TODO: Allow results on CPU
        Args:
            y: [B, V, H, W]
            pred: [B, V, H, W]
            vars: list of variable names
            lat: H
        """

        squared_error = (pred - y) ** 2  # [B, V, H, W]

        if pred.get_device() == -1:
            squared_error = self.area_weights.cpu()*squared_error
        else:
            squared_error = self.area_weights*squared_error

        loss_dict = {}

        if len(self.masking_values_dict) == 0:

            per_var_mse = torch.mean(squared_error, dim=(0, -2, -1))
            loss_dict['mse'] = 0

            sum_var_weights = 0
            for i, var in enumerate(var_names):
                loss_dict[f"mse/{var}"] = per_var_mse[i]
                var_weight = self.variable_weights.get(var, 1)
                sum_var_weights += var_weight
                loss_dict['mse'] += per_var_mse[i] * var_weight

            loss_dict['mse'] = loss_dict['mse'] / sum_var_weights

        else:
            raise NotImplementedError("Masking not implemented for mse")

        return loss_dict

    def plain_mse(self, pred, y, var_names):
        loss_dict = {}
        loss_dict['plain_mse'] = torch.nn.functional.mse_loss(pred, y)

        return loss_dict


    def rmse(self, pred, y, var_names):
        """Latitude weighted root mean squared error

        Args:
            y: [B, V, H, W]
            pred: [B, V, H, W]
            vars: list of variable names
            lat: H
        """

        squared_error = (pred - y) ** 2  # [N, C, H, W]

        if pred.get_device() ==-1:
            squared_error = self.area_weights.cpu()*squared_error
        else:
            squared_error = self.area_weights*squared_error

        loss_dict = {}

        if len(self.masking_values_dict) == 0:
            

            per_var_rmse = torch.sqrt(torch.mean(squared_error, dim=(-2, -1)))
            per_var_rmse = per_var_rmse.mean(dim=0)
            loss_dict['rmse'] = per_var_rmse.mean()

            sum_var_weights = 0
            for i, var in enumerate(var_names):
                loss_dict[f"{'rmse'}/{var}"] = per_var_rmse[i]
                var_weight = self.variable_weights.get(var, 1)
                loss_dict['rmse'] += per_var_rmse[i] * var_weight
                sum_var_weights += var_weight


            loss_dict['rmse'] = loss_dict['rmse'] / sum_var_weights

        else:
            raise NotImplementedError("Masking not implemented for lat_weighted_mse")

        return loss_dict