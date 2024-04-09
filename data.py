from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split

class AdataDataset(Dataset):
    """
    A custom dataset class for representing gene data.

    Args:
        genes (numpy.ndarray): Array of gene data.
        batches (numpy.ndarray): Array of batch information.

    Attributes:
        genes (numpy.ndarray): Array of gene data.
        batches (numpy.ndarray): Array of batch information.
    """

    def __init__(self, genes, batches):
        self.genes = genes
        self.batches = batches

    def __len__(self):
        return self.genes.shape[0]

    def __getitem__(self, idx):
        """
        Get a specific item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the gene data and batch information.
                - "X" (numpy.ndarray): Gene data at the specified index.
                - "batch" (numpy.ndarray): Batch information at the specified index.
        """
        batch = {
            "X": self.genes[idx],
            "batch": self.batches[idx],
        }
        return batch

class AdataDataModule(pl.LightningDataModule):
    def __init__(self, adata, batch_size=2048, val_split=0.1, test_split=0.1):
        """
        LightningDataModule for handling data loading and processing for AdataDataset.

        Args:
            adata (AnnData): Annotated data object containing gene expression data.
            batch_size (int): Number of samples per batch (default: 2048).
            val_split (float): Fraction of data to be used for validation (default: 0.1).
            test_split (float): Fraction of data to be used for testing (default: 0.1).
        """
        super().__init__()
        self.adata = adata
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        """
        Prepare the train, validation, and test datasets.

        Args:
            stage (str, optional): The current stage (e.g., 'fit', 'validate', 'test'). Defaults to None.
        """
        train_genes, val_test_genes, train_batches, val_test_batches = train_test_split(
            self.adata.X, self.adata.obs['scRNASeq_sample_ID'],
            test_size=self.val_split + self.test_split, random_state=42
        )
        val_genes, test_genes, val_batches, test_batches = train_test_split(
            val_test_genes, val_test_batches,
            test_size=self.test_split / (self.val_split + self.test_split), random_state=42
        )

        self.train_dataset = AdataDataset(train_genes.todense(), train_batches)
        self.val_dataset = AdataDataset(val_genes.todense(), val_batches)
        self.test_dataset = AdataDataset(test_genes.todense(), test_batches)

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def __iter__(self):
        """
        Iterates over the train, validation, and test dataloaders.

        Yields:
            torch.utils.data.DataLoader: The train, validation, and test dataloaders.
        """
        yield from (self.train_dataloader(), self.val_dataloader(), self.test_dataloader())