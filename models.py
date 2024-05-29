import torch
import torch.nn as nn
import lightning.pytorch as pl
import abc
from typing import List, Optional, Callable, Dict
import gc
from torch.distributions import Bernoulli, ContinuousBernoulli
import torch.nn.functional as F
from torchmetrics import ExplainedVariance, MeanSquaredError, MetricCollection


class MLP(torch.nn.Sequential):
    """
    This class implements the multi-layer perceptron (MLP) module.
    It uses torch.nn.Sequential to make the forward call sequentially.
    Implementation slightly adapted from https://pytorch.org/vision/main/generated/torchvision.ops.MLP.html
    (removed Dropout from last layer + log_api_usage call)

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear
        layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of
         the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer wont be
         used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place.
        Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        dropout: float = 0.0,
        final_activation: Optional[Callable[..., torch.nn.Module]] = None,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim, eps=0.001))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        # the last layer should not have dropout
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        if final_activation is not None:
            layers.append(final_activation())

        super().__init__(*layers)


class BaseAutoEncoder(pl.LightningModule, abc.ABC):
    """
    Base class for AutoEncoder models in PyTorch Lightning.

    Args:
        gene_dim (int): Dimension of the gene.
        batch_size (int): Batch size.
        reconst_loss (str, optional): Reconstruction loss function. Defaults to 'mse'.
        learning_rate (float, optional): Learning rate. Defaults to 0.005.
        weight_decay (float, optional): Weight decay. Defaults to 0.1.
        optimizer (Callable[..., torch.optim.Optimizer], optional): Optimizer class. Defaults to torch.optim.AdamW.
        lr_scheduler (Callable, optional): Learning rate scheduler. Defaults to None.
        lr_scheduler_kwargs (Dict, optional): Additional arguments for the learning rate scheduler. Defaults to None.
        gc_frequency (int, optional): Frequency of garbage collection. Defaults to 1.
        automatic_optimization (bool, optional): Whether to use automatic optimization. Defaults to True.
        supervised_subset (Optional[int], optional): Subset size for supervised training. Defaults to None.
    """

    autoencoder: nn.Module  # autoencoder mapping von gene_dim to gene_dim

    def __init__(
        self,
        gene_dim: int,
        batch_size: int,
        reconst_loss: str = "mse",
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        gc_frequency: int = 1,
        automatic_optimization: bool = True,
    ):
        super(BaseAutoEncoder, self).__init__()

        self.automatic_optimization = automatic_optimization

        self.gene_dim = gene_dim
        self.batch_size = batch_size
        self.gc_freq = gc_frequency

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optim = optimizer
        self.reconst_loss = reconst_loss
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        metrics = MetricCollection(
            {
                "explained_var_weighted": ExplainedVariance(
                    multioutput="variance_weighted"
                ),
                "explained_var_uniform": ExplainedVariance(
                    multioutput="uniform_average"
                ),
                "mse": MeanSquaredError(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _calc_reconstruction_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
    ):
        """
        Calculate the reconstruction loss.

        Args:
            preds (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target values.
            reduction (str, optional): Reduction method. Defaults to 'mean'.

        Returns:
            torch.Tensor: Reconstruction loss.
        """
        if self.reconst_loss == "continuous_bernoulli":
            loss = -ContinuousBernoulli(probs=preds).log_prob(targets)
            if reduction == "mean":
                loss = loss.mean()
            elif reduction == "sum":
                loss = loss.sum()
        elif self.reconst_loss == "bce":
            loss = F.binary_cross_entropy(preds, targets, reduction=reduction)
        elif self.reconst_loss == "mae":
            loss = F.l1_loss(preds, targets, reduction=reduction)
        else:
            loss = F.mse_loss(preds, targets, reduction=reduction)
        return loss

    @abc.abstractmethod
    def _step(self, batch):
        """
        Calculate predictions (int64 tensor) and loss.

        Args:
            batch: Input batch.

        Returns:
            tuple: Tuple containing predictions and loss.
        """
        pass

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Perform operations on the batch after it has been transferred to the device.

        Args:
            batch: Input batch.
            dataloader_idx: Index of the dataloader.

        Returns:
            batch: Processed batch.
        """
        if isinstance(batch, dict):  # Case for MultiomicsDataloader
            return batch
        else:
            return batch[0]

    def forward(self, batch):
        """
        Forward pass of the autoencoder.

        Args:
            batch: Input batch.

        Returns:
            tuple: Tuple containing the latent representation and the reconstructed output.
        """
        x_in = batch["X"]
        x_latent = self.encoder(x_in)
        x_reconst = self.decoder(x_latent)
        return x_latent, x_reconst

    def on_train_epoch_end(self) -> None:
        """
        Perform operations at the end of each training epoch.
        """
        gc.collect()

    def on_validation_epoch_end(self) -> None:
        """
        Perform operations at the end of each validation epoch.
        """
        gc.collect()

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: Configuration for the optimizer and learning rate scheduler.
        """
        optimizer_config = {
            "optimizer": self.optim(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        }
        if self.lr_scheduler is not None:
            lr_scheduler_kwargs = (
                {} if self.lr_scheduler_kwargs is None else self.lr_scheduler_kwargs
            )
            interval = lr_scheduler_kwargs.pop("interval", "epoch")
            monitor = lr_scheduler_kwargs.pop("monitor", "val_loss_epoch")
            frequency = lr_scheduler_kwargs.pop("frequency", 1)
            scheduler = self.lr_scheduler(
                optimizer_config["optimizer"], **lr_scheduler_kwargs
            )
            optimizer_config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": interval,
                "monitor": monitor,
                "frequency": frequency,
            }

        return optimizer_config


class MLPAutoEncoder(BaseAutoEncoder):
    def __init__(
        self,
        # fixed params
        gene_dim: int,
        units_encoder: List[int],
        units_decoder: List[int],
        batch_size: int,
        reconstruction_loss: str = "mse",
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        dropout: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        output_activation: Callable[[], torch.nn.Module] = nn.Sigmoid,
        activation: Callable[[], torch.nn.Module] = nn.SELU,
        # params for masking
        masking_rate: Optional[float] = None,
        masking_strategy: Optional[str] = None,  # 'random', 'gene_program'
    ):
        # check input
        assert 0.0 <= dropout <= 1.0
        assert reconstruction_loss in ["mse", "mae", "continuous_bernoulli", "bce"]
        if reconstruction_loss in ["continuous_bernoulli", "bce"]:
            assert output_activation == nn.Sigmoid

        self.batch_size = batch_size

        super(MLPAutoEncoder, self).__init__(
            gene_dim=gene_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

        self.encoder = MLP(
            in_channels=gene_dim,
            hidden_channels=units_encoder,
            activation_layer=activation,
            inplace=False,
            dropout=dropout,
        )
        # Define decoder network
        self.decoder = nn.Sequential(
            MLP(
                in_channels=units_encoder[-1],
                hidden_channels=units_decoder + [gene_dim],
                # norm_layer=_get_norm_layer(batch_norm=batch_norm, layer_norm=layer_norm),
                activation_layer=activation,
                inplace=False,
                dropout=dropout,
            ),
            output_activation(),
        )

        self.predict_bottleneck = False

        metrics = MetricCollection(
            {
                "explained_var_weighted": ExplainedVariance(
                    multioutput="variance_weighted"
                ),
                "explained_var_uniform": ExplainedVariance(
                    multioutput="uniform_average"
                ),
                "mse": MeanSquaredError(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # masking
        self.masking_rate = masking_rate
        self.masking_strategy = masking_strategy

    def _step(self, batch, training=True):
        targets = batch["X"]
        inputs = batch["X"]

        if self.masking_rate and self.masking_strategy == "random":
            mask = (
                Bernoulli(probs=1.0 - self.masking_rate)
                .sample(targets.size())
                .to(targets.device)
            )
            # upscale inputs to compensate for masking and convert to same device
            masked_inputs = 1.0 / (1.0 - self.masking_rate) * (inputs * mask)
            x_latent, x_reconst = self(masked_inputs)
            # calculate masked loss on masked part only
            inv_mask = torch.abs(torch.ones(mask.size()).to(targets.device) - mask)
            loss = (
                inv_mask
                * self._calc_reconstruction_loss(x_reconst, targets, reduction="none")
            ).mean()

        else:
            x_latent, x_reconst = self(inputs.to(targets.device))
            loss = self._calc_reconstruction_loss(x_reconst, targets, reduction="mean")

        return x_reconst, loss

    def predict_embedding(self, batch):
        return self.encoder(batch["X"])

    def forward(self, x_in):
        x_latent = self.encoder(x_in)
        x_reconst = self.decoder(x_latent)
        return x_latent, x_reconst

    def training_step(self, batch, batch_idx):
        x_reconst, loss = self._step(batch)
        self.log_dict(
            self.train_metrics(x_reconst, batch["X"]), on_epoch=True, on_step=True
        )
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        x_reconst, loss = self._step(batch, training=False)
        self.log_dict(self.val_metrics(x_reconst, batch["X"]))
        self.log("val_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def test_step(self, batch, batch_idx):
        x_reconst, loss = self._step(batch, training=False)
        metrics = self.test_metrics(x_reconst, batch["X"])
        self.log_dict(metrics)
        self.log("test_loss", loss)
        if batch_idx % self.gc_freq == 0:
            gc.collect()
        return metrics

    def predict_cell_types(self, x: torch.Tensor):
        return F.softmax(self(x)[0], dim=1)

    def predict_step(
        self, batch, batch_idx, dataloader_idx=None, predict_embedding=False
    ):
        if batch_idx % self.gc_freq == 0:
            gc.collect()

        if predict_embedding:
            return self.encoder(batch["X"]).detach()
        else:
            x_reconst, loss = self._step(batch, training=False)
            return x_reconst, batch["X"]

    def get_input(self, batch):
        return batch["X"]