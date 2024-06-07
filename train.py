import argparse
import copy
import logging
import os
import shutil
from typing import Any, Optional, Type

import gin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import sklearn.model_selection
import sklearn.neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import tqdm

from dataset import Vcc2018, get_dataloader
from model import DnsmosPro


parser = argparse.ArgumentParser(description='Gin and save path.')
parser.add_argument(
    '--gin_path',
    type=str,
    help='Path to the gin-config.',
    default='configs/tot.gin'
)
parser.add_argument(
    '--save_path',
    type=str,
    help='Path to directory storing results.',
)
args = parser.parse_args()


@gin.configurable
class TrainingLoop:
    """The training loop which trains and evaluates a model."""

    def __init__(
        self,
        *,
        model: nn.Module = DnsmosPro,
        save_path: str = '',
        loss_type: str = 'gnll',
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        weight_decay: float = 0.0,
        dataset_cls: Type[Dataset] = Vcc2018,
        num_epochs: int = 500,
        learning_rate: float = 1e-4,
        batch_size_train: int = 64,
        frame_level_loss: bool = False,
    ):
        """Initializes the instance.
        
        Args:
            model: The nn.Module model. Expected to take (B, 1, T, F) as input,
                and output (B, S), where B is the batch size, T is the time bins,
                F is the frequency bins, and S is the score.
            save_path: Path to log directory.
            loss_type: Type of loss, 'mse', 'mae', and 'gnll' are supported.
            optimizer: The optimizer.
            weight_decay: Weight decay of the parameters.
            dataset_cls: The dataset class. Expected to have the parameter
                `valid`, which can take the values 'train', 'val', and 'test'.
                Returns a `Dataset` object.
            num_epochs: Number of training epochs.
            learning_rate: The learning rate.
            batch_size_train: Batch size of train set.
            frame_level_loss: If frame level loss and predictions is used.
        """
        # Setup logging and paths.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print('New directory added!')
        log_path = os.path.join(save_path, 'train.log')
        self._save_path = save_path
        logging.basicConfig(filename=log_path, level=logging.INFO)

        # Datasets.
        train_dataset = dataset_cls(valid='train')
        valid_dataset = dataset_cls(valid='val')
        test_dataset = dataset_cls(valid='test')
        logging.info(f'Num train speech clips: {len(train_dataset)}')
        logging.info(f'Num val speech clips: {len(valid_dataset)}')
        logging.info(f'Num test speech clips: {len(test_dataset)}')
        self._label_type = train_dataset.label_type
        self._train_loader = get_dataloader(
            dataset=train_dataset,
            batch_size=batch_size_train
        )
        self._valid_loader = get_dataloader(
            dataset=valid_dataset,
            batch_size=1
        )
        self._test_loader = get_dataloader(
            dataset=test_dataset,
            batch_size=1
        )

        # Model and optimizers.
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'Device={self._device}')
        self._model = model().to(self._device)
        self._best_pcc = -1
        # TODO: Explore some learning rate scheduler.
        self._optimizer = optimizer(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self._optimizer.zero_grad()
        if loss_type == 'mse':
            self._loss_fn = F.mse_loss
        elif loss_type == 'mae':
            self._loss_fn = F.l1_loss
        elif loss_type == 'gnll':
            self._loss_fn = F.gaussian_nll_loss
        else:
            raise ValueError(f'Loss {loss_type} not supported.')

        self._loss_type = loss_type
        self._all_loss = []
        self._epoch = 0
        self._num_epochs = num_epochs
        self._frame_level_loss = frame_level_loss
    
    @property
    def save_path(self):
        """The path to the log directory."""
        return self._save_path
            
    def _train_once(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Performs forward and backward pass on batch.

        Args:
            batch: The batch consisting of the spectrograms and labels.
        """
        specs, labels = batch
        specs = specs.to(self._device)
        specs = specs.unsqueeze(1)  # Shape (B, 1, T, F).
        labels = labels.to(self._device)

        # Forward
        predictions = self._model(specs)
        if self._frame_level_loss:
            predictions = predictions.squeeze()
            mean_predictions = torch.mean(predictions, dim=-1)
            labels_repeated = labels.unsqueeze(1).repeat(1, predictions.shape[1])
            loss = self._loss_fn(labels, mean_predictions) + self._loss_fn(labels_repeated, predictions)
        else:
            if self._loss_type == 'gnll':
                mean = predictions[:, 0]
                var = predictions[:, 1]
                loss = self._loss_fn(mean, labels, var)
            else:
                predictions = predictions.squeeze(-1)
                loss = self._loss_fn(labels, predictions)
                
        # Backwards
        loss.backward()
        self._all_loss.append(loss.item())
        del loss

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5)
        self._optimizer.step()
        self._optimizer.zero_grad()
    
    def train(self, valid_each_epoch: bool = True) -> None:
        """Trains the model on the train data `self._num_epochs` number of epochs.
        
        Args:
            valid_each_epoch: If to compute the validation performance.
        """
        self._model.train()
        while self._epoch <= self._num_epochs:
            self._all_loss = list()
            #if self._epoch == 20:
            #    self._optimizer_momentum.set_alpha(0.999)
            for batch in tqdm.tqdm(
                self._train_loader,
                ncols=0,
                desc="Train",
                unit=" step"
            ):
                self._train_once(batch)

            average_loss = torch.FloatTensor(self._all_loss).mean().item()
            logging.info(f'Average loss={average_loss}')

            if valid_each_epoch:
                _ = self.valid()
            self._epoch += 1

    def _evaluate(self, dataloader: Any, prefix: str):
        """Evaluates the model on the data based on quality prediction."""
        self._model.eval()
        predictions, labels = [], []
        # TODO: Consider adding support of system level performance.
        for i, batch in enumerate(tqdm.tqdm(
            dataloader,
            ncols=0,
            desc=prefix,
            unit=' step'
        )):
            wav, label = batch
            wav = wav.to(self._device)
            wav = wav.unsqueeze(1) # shape (batch, 1, seq_len, [dim feature])

            with torch.no_grad():
                try:
                    prediction = self._model(wav) # shape (batch, 1)
                    if self._frame_level_loss:
                        prediction = torch.mean(prediction.squeeze(-1), dim=-1).unsqueeze(-1)
                    if self._loss_type == 'gnll':
                        prediction = prediction[:, 0]
                    else:
                        prediction = prediction.squeeze(-1) # shape (batch)

                    prediction = prediction.cpu().detach().numpy()
                    predictions.extend(prediction.tolist())
                    labels.extend(label.tolist())
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        logging.error(f'[Runner] - CUDA out of memory at step {global_step}')
                        with torch.cuda.device(self._device):
                            torch.cuda.empty_cache()
                    else:
                        raise

        predictions = np.array(predictions)
        labels = np.array(labels)
        utt_mse=np.mean((labels-predictions)**2)
        utt_pcc=np.corrcoef(labels, predictions)[0][1]
        utt_srcc=scipy.stats.spearmanr(labels, predictions)[0]
        if utt_pcc > self._best_pcc:
            self._best_pcc = utt_pcc
            self.save_model('model_best.pt')

        logging.info(
            f"\n[{prefix}][{self._epoch}][UTT][ MSE = {utt_mse:.4f} | LCC = {utt_pcc:.4f} | SRCC = {utt_srcc:.4f} ]"
        )
        self._model.train()
        return predictions, labels
    
    def valid(self):
        """Evaluates the model on validation data."""
        return self._evaluate(self._valid_loader, 'Valid')
    
    def test(self, plot: bool = False):
        """Evaluates the model on test data."""
        self._model = torch.jit.load(os.path.join(self._save_path, 'model_best.pt')).to(self._device)
        predictions, labels = self._evaluate(self._test_loader, 'Test')
        if plot:
            plt.scatter(labels, predictions)
            plt.xlim([0.9, 5.1])
            plt.ylim([0.9, 5.1])
            plt.xlabel(self._label_type)
            plt.ylabel('Predictions')
            plt.title('Test data predictions vs targets')
            plt.savefig(os.path.join(self._save_path, 'test_scatter.png'))
        return predictions, labels

    def save_model(self, model_name: str = 'model.pt'):
        """Saves the model."""
        model_scripted = torch.jit.script(self._model)
        model_scripted.save(os.path.join(self._save_path, model_name))


def main():
    """Main."""
    gin.external_configurable(
            torch.nn.modules.activation.ReLU,
            module='torch.nn.modules.activation'
            )
    gin.external_configurable(
            torch.nn.modules.activation.SiLU,
            module='torch.nn.modules.activation'
            )
    gin.parse_config_file(args.gin_path)
    train_loop = TrainingLoop(save_path=args.save_path)
    new_gin_path = os.path.join(train_loop.save_path, 'config.gin')
    shutil.copyfile(args.gin_path, new_gin_path)
    train_loop.train()
    train_loop.test(plot=True)


if __name__ == '__main__':
    main()
