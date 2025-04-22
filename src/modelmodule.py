
import os

import torch
from pytorch_lightning import LightningModule
from torch.utils.checkpoint import checkpoint

from models.base_model import NetArchitecture
from utils.train_utils import LinearWarmupCosineAnnealingLR
from metrics import MetricsModule


class MultistepForecastModule(LightningModule):
    """
    A PyTorch Lightning module for training a multistep forecast model.
    This module is designed to work with any model architecture inheriting from NetArchitecture.

    Args:
        net: The neural network architecture to be used for forecasting.
        metrics_module: The module containing the metrics for evaluation.
        pretrained_path: Path to a pretrained model checkpoint. Defaults to None.
        optimizer: The optimizer to use as string. Will use optimizer from torch.optim. Defaults to 'Adam'.
        optimizer_args: Additional arguments for the optimizer. Defaults to {}.
        schedule: The learning rate schedule to use. Options are None, warmup_cosine, cosine. Defaults to None.
        schedule_args: Additional arguments for the learning rate schedule. Defaults to {}.
        gradient_checkpointing: Whether to use gradient checkpointing. Defaults to False.
    """

    def __init__(
        self,
        net: NetArchitecture,
        metrics_module: MetricsModule,
        pretrained_path: str = None,
        optimizer: str = 'Adam',
        optimizer_args: dict = {},
        schedule: str = None,
        schedule_args: dict = {},
        gradient_checkpointing: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.net = net
        self.metrics_module = metrics_module
        self.gradient_checkpointing = gradient_checkpointing
        self.processed_train_samples = 0

        if pretrained_path is not None:
            self.load_pretrained_weights(os.path.join(
                os.environ['RESULTS_DIR'], pretrained_path))

    def load_pretrained_weights(self, pretrained_path):

        checkpoint = torch.load(
            pretrained_path, map_location=torch.device("cpu"), weights_only=True)
        checkpoint_weights = checkpoint['state_dict']
        checkpoint_weights = {
            k.replace("net.", ""): v for k, v in checkpoint_weights.items()}
        # print(list(checkpoint_weights.keys()))
        msg = self.net.load_state_dict(checkpoint_weights, strict=False)

        print(msg)

    def set_datamodule(self, datamodule):
        '''
        Pass parameters from the dataset to the model for training.

        ToDo: Solve this in a more elegant way with argument linking in the cli.
        '''
        self.datamodule = datamodule

    def setup(self, stage=None):
        '''
        Setup the model for training or evaluation. This is called by PyTorch Lightning.
        It is called once for each stage (train, val, test) and is used to set up the model for training or evaluation.

        Parameters from the datamodule are extracted

        '''

        dataset = self.datamodule.get_dataset_for_statistics()

        self.norm_init = dataset.norm_init
        self.norm_forcing = dataset.norm_forcing
        self.norm_target = dataset.norm_target

        self.denorm_preds = dataset.denorm_preds

        self.var_names_in = tuple(dataset.var_names_in)  # Tuple for caching
        self.var_names_out = tuple(dataset.var_names_out)
        self.var_names_prognostics = tuple(dataset.var_names_prognostics)

        self.constant_vars_idx = dataset.constant_vars_idx
        self.prognostic_vars_idx_input = dataset.prognostic_vars_idx_input
        self.prognostic_vars_idx_output = dataset.prognostic_vars_idx_output

        self.metrics_module.set_lat(dataset.get_lat(), device=self.device)

    def training_step(self, batch: dict, batch_idx: int):
        """
        Training step for the model. This function is called by PyTorch Lightning during training
        Batch must contain the following keys:
        - initial_condition: The initial condition for the model. Shape: (B, C, H, W)
        - forcing: The forcing for the model. Shape: (B, steps-1, C2, H, W)
        - target: The target for the model. Shape: (B, steps, C3, H, W)
        - dates: The dates for the model. Shape: (B, steps)
        - kwargs: Additional keyword arguments for the model. These are passed to the model directly.
        """

        initial_condition = batch.pop('initial_condition')
        forcings = batch.pop('forcing')
        y = batch.pop('target')
        # Necessary, since all remaining elements in batch are directly passed to model
        dates = batch.pop('dates')

        initial_condition = self.norm_init(initial_condition)
        y = self.norm_target(y)

        if y.shape[1] > 1:
            assert forcings.shape[1]+1 == y.shape[1]
            forcings = self.norm_forcing(forcings)

        multisteploss = 0

        # Check for NaN or Inf values in input and skip the step
        if torch.isfinite(initial_condition).all():

            for i in range(y.shape[1]):

                if i == 0:
                    x = initial_condition
                else:
                    x = torch.cat([initial_condition[:, self.constant_vars_idx],
                                  forcings[:, i-1], preds[:, self.prognostic_vars_idx_output]], dim=1)

                if self.gradient_checkpointing:
                    if i == 0:
                        # This is necessary for gradient checkpointing to work, maybe there is a better workaround
                        x.requires_grad = True
                    preds = checkpoint(self.net, x, self.var_names_in, self.var_names_out,
                                       self.var_names_prognostics, use_reentrant=True)
                else:
                    preds = self.net.forward(x, var_names_in=self.var_names_in, var_names_out=self.var_names_out,
                                             var_names_prognostics=self.var_names_prognostics, **batch)  # B, prognostics+diagnostics, H, W

                # Compute loss
                target = y[:, i]
                step_loss = 0

                # Compute loss for each loss metric
                for (metric_name, metric_fn, metric_weight) in self.metrics_module.train_metrics:
                    metric_dict = metric_fn(preds, target, self.var_names_out)
                    step_loss += metric_weight*metric_dict[metric_name]

                    metric_dict = {"train/" + str((i+1)*self.net.lead_time_hours) +
                                   "_hours/norm_"+var: value for var, value in metric_dict.items()}

                    self.log_dict(
                        metric_dict,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        batch_size=initial_condition.shape[0],
                    )

                multisteploss += step_loss

            multisteploss = multisteploss/y.shape[1]

            self.processed_train_samples += initial_condition.shape[0]
            self.log(
                "processed_train_samples",
                self.processed_train_samples,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
                batch_size=initial_condition.shape[0],
            )
            return multisteploss
        else:
            print(
                f"NaN or Inf values in input for initial conditions {dates[:,0]}")
            return None

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Evaluation step for the model. This function is called by PyTorch Lightning during validation
        """

        self.eval_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int):
        """
        Evaluation step for the model. This function is called by PyTorch Lightning during testing
        """

        self.eval_step(batch, "test")

    def eval_step(self, batch, log_prefix):
        """
        Evaluation step for the model.
        Performs a rollout during validation/testing and logs metrics for each step to WandB.
        """

        initial_condition = batch.pop('initial_condition')
        forcings = batch.pop('forcing')
        y = batch.pop('target')
        # Necessary, since all remaining elements in batch are directly passed to model
        dates = batch.pop('dates')

        # We could save memory by not storing the not normalized targets. We could then normalize everything in the dataloader
        initial_condition = self.norm_init(initial_condition)
        y_norm = self.norm_target(y)

        if y.shape[1] > 1:
            assert forcings.shape[1] + \
                1 == y.shape[1], f"Number of forcings {forcings.shape[1]} does not match number of steps {y.shape[1]}"
            forcings = self.norm_forcing(forcings)

        all_metrics_aggregated = {}
        all_metrics = {}
        for i in range(y.shape[1]):

            if i == 0:
                input = initial_condition
            else:
                input = torch.cat([initial_condition[:, self.constant_vars_idx],
                                  forcings[:, i-1], preds_norm[:, self.prognostic_vars_idx_output]], dim=1)

            preds_norm = self.net.forward(input, var_names_in=self.var_names_in, var_names_out=self.var_names_out,
                                          var_names_prognostics=self.var_names_prognostics, **batch)  # B, prognostics+diagnostics, H, W

            normalized_metric_dicts = [
                m(preds_norm, y_norm[:, i], self.var_names_out) for m in self.metrics_module.eval_metrics]

            preds = self.denorm_preds(preds_norm.to(dtype=torch.float32))

            metrics_dict = [m(preds, y[:, i], self.var_names_out)
                            for m in self.metrics_module.eval_metrics]

            # Collect computed metrics in single dictionary

            for d in normalized_metric_dicts:
                for metric_name in d.keys():
                    metric_value = d[metric_name]
                    all_metrics[f'{log_prefix}_{(i+1)*self.net.lead_time_hours}_hours_norm_{metric_name}'] = metric_value

                    all_metrics_aggregated[f'{log_prefix}_agg_norm_{metric_name}'] = all_metrics_aggregated.get(
                        f'{log_prefix}_agg_norm_{metric_name}', 0) + metric_value

            for d in metrics_dict:
                for metric_name in d.keys():
                    metric_value = d[metric_name]
                    all_metrics[f'{log_prefix}_{(i+1)*self.net.lead_time_hours}_hours_{metric_name}'] = metric_value
                    all_metrics_aggregated[f'{log_prefix}_agg_{metric_name}'] = all_metrics_aggregated.get(
                        f'{log_prefix}/{metric_name}_agg', 0) + metric_value

        # Log metrics
        self.log_dict(
            all_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=initial_condition.shape[0],
        )

        # Log metrics aggregated over all steps
        for var in all_metrics_aggregated.keys():
            self.log(
                var,
                all_metrics_aggregated[var]/(y.shape[1]),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=initial_condition.shape[0],
            )

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for the model. Called by PyTorch Lightning.

        """
        optimizer = {}

        if 'decay' in self.hparams.optimizer_args:
            decay_value = self.hparams.optimizer_args.pop('decay')
            if self.hparams.opt > 0:
                decay = []
                no_decay = []
            for name, m in self.named_parameters():
                if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                    no_decay.append(m)
                else:
                    decay.append(m)

            optimizer['optimizer'] = eval(f'torch.optim.{self.hparams.optimizer}')(
                [
                    {'params': decay, 'decay': decay_value,
                        **self.hparams.optimizer_args},
                    {'params': no_decay, **self.hparams.optimizer_args}
                ])

        else:
            optimizer['optimizer'] = eval(f'torch.optim.{self.hparams.optimizer}')(
                self.net.parameters(), **self.hparams.optimizer_args)

        if self.hparams.schedule == 'warmup_cosine':
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer['optimizer'],
                **self.hparams.schedule_args
            )
            optimizer["lr_scheduler"] = {
                "scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        elif 'cosine':
            optimizer["lr_scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer['optimizer'], **self.hparams.schedule_args)

        else:
            print("No scheduler specified")

        return optimizer
