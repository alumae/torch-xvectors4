from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import sys
import os
import logging
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser

from lightning.pytorch.core import LightningModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback

import data
from model import SpeechClassificationModel
from lightning.pytorch.callbacks import BasePredictionWriter


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--load-model", type=str)
        parser.link_arguments("model.label2id", "data.label2id", apply_on='instantiate')

    # def instantiate_classes(self) -> None:
        
    #     """Instantiates the classes and sets their attributes."""
    #     #breakpoint()    
    #     load_model_arg = self.config[self.subcommand]["load_model"]
    #     if load_model_arg is not None:
    #         self.model = self.model_class.load_from_checkpoint(checkpoint_path=load_model_arg)
    #         self.config[self.subcommand].pop("model")

    #     #breakpoint()    
    #     self.config_init = self.parser.instantiate_classes(self.config)
    #     self.datamodule = self._get(self.config_init, "data")
        
    #     self.model = self._get(self.config_init, "model")
    #     self._add_configure_optimizers_method_to_model(self.subcommand)
    #     self.trainer = self.instantiate_trainer()

class XVectorWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        
        #torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))
        from kaldiio import WriteHelper
        write_helper = WriteHelper(f'ark,scp:{self.output_dir}/xvector.{trainer.global_rank+1}.ark,{self.output_dir}/xvector.{trainer.global_rank+1}.scp')
        for batch_predictions in predictions:
            for utt_id, xvector in batch_predictions.items():
                write_helper(utt_id, xvector.cpu().numpy().flatten())
        write_helper.close()




if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    cli = MyLightningCLI(
        SpeechClassificationModel, data.DataModule, seed_everything_default=1234, save_config_kwargs={"overwrite": True}, run=True
    )
