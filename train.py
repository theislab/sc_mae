import lightning.pytorch as pl
import scanpy as sc
from data import AdataDataModule
from models import MLPAutoEncoder
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/lustre/groups/ml01/workspace/till.richter/patient_rep/combat_processed.h5ad",
    )
    parser.add_argument("--units_encoder", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--units_decoder", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--masking_strategy", type=str, default="random")
    parser.add_argument("--masking_rate", type=float, default=0.5)
    return parser.parse_args()


def train(
    adata,
    units_encoder=[512, 256, 128],
    units_decoder=[128, 256, 512],
    batch_size=2048,
    masking_strategy="random",
    masking_rate=0.5,
):
    """
    Train the MLPAutoEncoder model using the provided data.

    Args:
        adata (AnnData): The input data for training.
        units_encoder (list, optional): List of integers specifying the number of units in each encoder layer. Defaults to [512, 256, 128].
        units_decoder (list, optional): List of integers specifying the number of units in each decoder layer. Defaults to [128, 256, 512].
        batch_size (int, optional): The batch size for training. Defaults to 2048.
        masking_strategy (str, optional): The masking strategy to be used during training. Defaults to "random".
        masking_rate (float, optional): The rate of masking to be applied during training. Defaults to 0.5.

    Returns:
        None
    """
    model = MLPAutoEncoder(
        gene_dim=adata.shape[1],
        units_encoder=units_encoder,
        units_decoder=units_decoder,
        batch_size=batch_size,
        masking_strategy=masking_strategy,
        masking_rate=masking_rate,
    )
    datamodule = AdataDataModule(adata=adata, batch_size=batch_size)

    trainer = pl.Trainer(
        max_epochs=1000,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    args = parse_args()
    adata = sc.read_h5ad(args.data_path)
    train(
        adata,
        units_encoder=args.units_encoder,
        units_decoder=args.units_decoder,
        batch_size=args.batch_size,
        masking_strategy=args.masking_strategy,
        masking_rate=args.masking_rate,
    )
