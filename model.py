import os
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from collections import OrderedDict
import functools
import operator
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch import optim

from lightning.pytorch.core import LightningModule
from torchmetrics import Accuracy
from kaldiio import WriteHelper
from transformers import get_linear_schedule_with_warmup

from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import get_seqs_and_padding_mask
from pathlib import Path
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model
from fairseq2.data.data_pipeline import Collater
from fairseq2.nn.lora import LoRAConfig
from fairseq2.nn.lora import (
    freeze_non_lora,
    merge_lora,
    unmerge_lora,
    unwrap_lora,
    wrap_lora,
)

from loss import SoftmaxLoss
from pooling import *

EPSILON = torch.tensor(torch.finfo(torch.float).eps)




class SpeechClassificationModel(LightningModule):

    def __init__(self, 
                 wav2vec2_model: str = None,
                 w2vbert2_model:str = None,
                 pooling: str = "global-mha",
                 pooling_attention_hidden_dim: int = 64,
                 pre_pooling_hidden_dim: int = -1,
                 freeze_backbone_steps: int = 0,
                 warmup_steps: int = 100,
                 entropy_regularization: float = 0.0,
                 backbone_lr_scale: float = 0.01,
                 optimizer_name: str = "adamw",
                 learning_rate: float = 0.0005,
                 weight_decay:float = 0.0,
                 hidden_dim: int = 512,                 
                 label_map_file: str = None,       
                 load_pretrained_model: str = None,    
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # For dumping x-vectors
        self.write_helper = None

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = (torch.rand((8,  16000*3)), torch.ones(8).long() * 16000*3)

        if self.hparams.label_map_file is not None:
            self.hparams.label2id = {}
            for l in open(self.hparams.label_map_file):
                label, id, = l.split()
                self.hparams.label2id[label] = int(id)
            
        else:
            raise Exception("Label file must be specified when training a new model")
    
        self.label2id = self.hparams.label2id

        #del self.hparams.label2id
        #print("Model label2id:" , self.hparams.label2id)
        self.__build_model()

    def setup(self, stage: str):
        # build model
        

        if self.hparams.load_pretrained_model is not None:
            logging.info(f"Loading pretrained model from {self.hparams.load_pretrained_model}")
            checkpoint = torch.load(self.hparams.load_pretrained_model, map_location='cpu')
            #SpeechClassificationModel._load_model_state(checkpoint)
            self.load_state_dict(checkpoint['state_dict'])
            # state_dict = checkpoint["state_dict"]
            # model_state_dict = self.state_dict()
            # is_changed = False
            # for k in state_dict:
            #     if k in model_state_dict:
            #         if state_dict[k].shape != model_state_dict[k].shape:
            #             logging.info(f"Skip loading parameter: {k}, "
            #                         f"required shape: {model_state_dict[k].shape}, "
            #                         f"loaded shape: {state_dict[k].shape}")
            #             state_dict[k] = model_state_dict[k].to(self.device)
            #             is_changed = True
            #     else:
            #         logging.info(f"Dropping parameter {k}")
            #         is_changed = True

            # if is_changed:
            #     checkpoint.pop("optimizer_states", None)
            
            # del checkpoint        

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        self.wav2vec2 = None
        if self.hparams.wav2vec2_model != None:
            from huggingface_wav2vec import HuggingFaceWav2Vec2
            self.wav2vec2 = HuggingFaceWav2Vec2(
                source=self.hparams.wav2vec2_model,
                output_norm=True,
                freeze=False,
                freeze_feature_extractor=True,
                save_path=".huggingface",
                is_conformer="conformer" in self.hparams.wav2vec2_model.lower())

            wavs = torch.randn((1,16000), dtype=torch.float)
            reps = self.wav2vec2(wavs)            
            self.encoder_output_dim = reps.shape[2]
        elif self.hparams.w2vbert2_model != None:
            #breakpoint()
            self.w2vbert2_fbank_converter = WaveformToFbankConverter(
                num_mel_bins=80,
                waveform_scale=2**15,
                channel_last=True,
                standardize=True,
                dtype=torch.float32,
                device=self.device
            )
            self.w2vbert2_collater = Collater(pad_value=1)
            self.w2vbert2_model  = load_conformer_shaw_model("conformer_shaw", device=self.device, dtype=torch.float32)

            lora_config = LoRAConfig(
                r=32,
                alpha=32.0,
                dropout_p=0.05,
                keys=[".*.self_attn.*(q_proj|v_proj)$"])
            self.w2vbert2_model = wrap_lora(self.w2vbert2_model, lora_config)
            freeze_non_lora(self.w2vbert2_model, unfreeze_bias="none")

            wavs = torch.randn((1,16000), dtype=torch.float)
            reps = self.compute_w2vbert_output(wavs, wav_lens=torch.tensor([1.0]))
            self.encoder_output_dim = reps.shape[2]

        else:
            raise Exception("Unknown model backbone")    

           
        pooling_map = {"stats": StatisticsPooling, 
                       "attentive-stats": AttentiveStatisticsPooling,
                       "lde" : LDEPooling, 
                       "mha": MultiHeadAttentionPooling, 
                       "global-mha": GlobalMultiHeadAttentionPooling,
                       "multires-mha": MultiResolutionMultiHeadAttentionPooling,
                       "ecapa-attentive-stats": EcapaAttentiveStatisticsPooling}
        


        pooling_input_dim = self.encoder_output_dim
        if self.hparams.pre_pooling_hidden_dim != -1:
            pre_pooling_layers = []
            pre_pooling_layers.append(nn.Conv1d(self.encoder_output_dim, self.hparams.pre_pooling_hidden_dim, kernel_size=self.hparams.pre_pooling_kernel_size, stride=self.hparams.pre_pooling_stride))
            pre_pooling_layers.append(nn.BatchNorm1d(self.hparams.pre_pooling_hidden_dim))
            pre_pooling_layers.append(nn.ReLU(inplace=True))
            self.pre_pooling_layers = nn.Sequential(*pre_pooling_layers)
            pooling_input_dim = self.hparams.pre_pooling_hidden_dim
        else:
            self.pre_pooling_layers = None


        self.pooling = pooling_map[self.hparams.pooling](input_dim=pooling_input_dim, hidden_size=self.hparams.pooling_attention_hidden_dim)

        
        post_pooling_layers = []

        post_pooling_layers.append(nn.Linear(self.pooling.get_output_dim(), self.hparams.hidden_dim))
        post_pooling_layers.append(nn.BatchNorm1d(self.hparams.hidden_dim))
        post_pooling_layers.append(nn.ReLU(inplace=True))

        post_pooling_layers.append(nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim))
        post_pooling_layers.append(nn.BatchNorm1d(self.hparams.hidden_dim))
        post_pooling_layers.append(nn.ReLU(inplace=True))
        self.post_pooling_layers = nn.Sequential(*post_pooling_layers)


        self.loss = SoftmaxLoss(input_dim=self.hparams.hidden_dim, 
                                        num_targets=len(self.hparams.label2id),
                                        entropy_regularization=self.hparams.entropy_regularization)
            

        #print(self.model)
        #from torchsummary import summary
        self.train_acc = Accuracy(task='multiclass', num_classes=len(self.hparams.label2id))
        self.valid_acc = Accuracy(task='multiclass', num_classes=len(self.hparams.label2id))
        
    

    def to(self, device):
        r = super().to(device)
        if self.hparams.w2vbert2_model != None:
            self.w2vbert2_fbank_converter = WaveformToFbankConverter(
                num_mel_bins=80,
                waveform_scale=2**15,
                channel_last=True,
                standardize=True,
                dtype=torch.float32,
                device=device
            )
        return r


    def compute_w2vbert_output(self, wavs, wav_lens):
        fbanks = []
        wav_lens = wav_lens / wav_lens.max()
        for i, wav in enumerate(wavs):
            wav = wav[0: int(len(wav) * wav_lens[i])]
            decoded_audio = {'sample_rate': 16000.0, 
                            'format': 65538, 
                            'waveform': wav.unsqueeze(1)}
            fbank = self.w2vbert2_fbank_converter(decoded_audio)["fbank"]
            fbanks.append(fbank)
        
        #breakpoint()
        src = self.w2vbert2_collater(fbanks)
        seqs, padding_mask = get_seqs_and_padding_mask(src)
        seqs, padding_mask = self.w2vbert2_model.encoder_frontend(seqs, padding_mask)
        seqs, padding_mask = self.w2vbert2_model.encoder(seqs, padding_mask)
        return seqs

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, wavs, wav_lens):
        pooling_output = self._forward_until_pooling(wavs, wav_lens)
        post_pooling_output = self.post_pooling_layers(pooling_output)
        return post_pooling_output

    def _forward_until_before_pooling(self, wavs, wav_lens):
        if self.wav2vec2 is not None:     
            if self.global_step < self.hparams.freeze_backbone_steps:
                with torch.no_grad():                
                    reps = self.wav2vec2(wavs)
            else:                    
                if self.global_step == self.hparams.freeze_backbone_steps and self.global_step > 0:
                    logging.info("Starting backbone training")
                reps = self.wav2vec2(wavs)
            #breakpoint()
            return reps.permute(0, 2, 1)
        elif self.w2vbert2_model is not None:
            #breakpoint()
            if self.global_step < self.hparams.freeze_backbone_steps:
                with torch.no_grad():                
                    reps = self.compute_w2vbert_output(wavs)
            else:                    
                if self.global_step == self.hparams.freeze_backbone_steps and self.global_step > 0:
                    logging.info("Starting backbone training")

                reps = self.compute_w2vbert_output(wavs, wav_lens)
            return reps.permute(0, 2, 1)

        else:
            raise Exception("not implemented")

    def _forward_until_pooling(self, wavs, wav_lens):
        pre_pooling_output = self._forward_until_before_pooling(wavs, wav_lens)
        if self.pre_pooling_layers:
            pre_pooling_output = self.pre_pooling_layers(pre_pooling_output)
        return self.pooling(pre_pooling_output).squeeze(2)

    def extract_xvectors(self, wavs, wav_lens, layer_index=1):
        pooling_output = self._forward_until_pooling(wavs, wav_lens)
        #breakpoint()
        return self.post_pooling_layers[0:layer_index](pooling_output)
        

    def extract_xvectors(self, wavs, wav_lens, layer_index=1):
        pooling_output = self._forward_until_pooling(wavs, wav_lens)
        #breakpoint()
        return self.post_pooling_layers[0:layer_index](pooling_output)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)
        # update learning rate
        self.lr_schedulers().step()


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)
        # update learning rate
        self.lr_schedulers().step()



    def configure_optimizers(self):
        params = list(self.named_parameters())

        def is_backbone(n): return ('wav2vec2' in n) or ('whisper_encoder' in n) #or ('w2vbert' in n)

        grouped_parameters = [
            {"params": [p for n, p in params if is_backbone(n)], 'lr': self.hparams.learning_rate * self.hparams.backbone_lr_scale},
            {"params": [p for n, p in params if not is_backbone(n)], 'lr': self.hparams.learning_rate},
        ]

        if self.hparams.optimizer_name == "adam":
            optimizer = torch.optim.Adam(grouped_parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)    
        elif self.hparams.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(grouped_parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)                        
        else:
            raise Exception(f"Unknown optimizer ({self.hparams.optimizer_name})")
        logging.info(f"Estimated stepping batches: {self.trainer.estimated_stepping_batches}")
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        #indexes = batch["index"]
        #logging.info(indexes)
        wav = batch["wav"]
        dur = batch["dur"]
        y = batch["label_id"]
        #y_hat = self.forward(wav, dur)
        y_hat = self.forward(wav, dur)

        loss_vector = self.loss(y_hat.unsqueeze(2), y)
        pred_vector = self.loss.get_posterior().squeeze(2).argmax(dim=1)

        loss_val = loss_vector.mean()

    
        lr  = torch.tensor(self.trainer.optimizers[0].param_groups[-1]['lr'], device=loss_val.device)
        self.log('train_loss', loss_val, prog_bar=True, sync_dist=True)
        self.log('lr', lr, prog_bar=True, rank_zero_only=True)
        self.log("train_acc", self.train_acc(pred_vector.cpu(), y.int().cpu()), prog_bar=True, sync_dist=True)
        
        return loss_val



    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        # Normal validation
        wav = batch["wav"]
        dur = batch["dur"]
        y = batch["label_id"]
        y_hat = self.forward(wav, dur)
        loss_vector = self.loss(y_hat.unsqueeze(2), y)
        pred_vector = self.loss.get_posterior().squeeze(2).argmax(dim=1)

        loss_val = loss_vector.mean()

        #breakpoint()
        # acc        
        self.log("val_acc", self.valid_acc(pred_vector.cpu(), y.int().cpu()), prog_bar=True, sync_dist=True)
        self.log('val_loss', loss_val, prog_bar=True, sync_dist=True)


    def test_step(self, batch, batch_idx):        
        wav = batch["wav"]
        dur = batch["dur"]
        utt_index = batch["index"]
        y = batch["label_id"]

        #breakpoint()
        if self.dump_xvectors_dir is not None:
            xvectors = self.extract_xvectors(wav, dur, self.xvector_layer_index)
        
            if self.write_helper is None and self.trainer.global_rank == 0:            
                with open(f"{self.dump_xvectors_dir}/architecture.txt", "w") as f:
                    print(self, file=f)

                self.write_helper = WriteHelper(f'ark,scp:{self.dump_xvectors_dir}/xvector.ark,{self.dump_xvectors_dir}/xvector.scp')

            
            for i in range(len(xvectors)):
                utt = self.trainer.test_dataloaders[0].dataset.utts[utt_index[i]]
                self.write_helper(utt, xvectors[i].cpu().numpy())
            output = OrderedDict({})
        elif self.dump_predictions is not None:
            if self.write_helper is None and self.trainer.global_rank == 0:            
                self.write_helper = WriteHelper(f'ark,scp:{self.dump_predictions}.ark,{self.dump_predictions}.scp')

            y_hat = self.forward(wav, dur)
            loss_val = self.loss(y_hat.unsqueeze(2), y)

            probs = F.log_softmax(self.loss.get_posterior(), dim=1)
            
            for i in range(len(probs)):
                utt = self.trainer.test_dataloaders[0].dataset.utts[utt_index[i]]
                self.write_helper(utt, probs[i].cpu().numpy().flatten())
            output = OrderedDict({})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def test_epoch_end(self, outputs):
        logging.info("Closing write helper...")
        if self.write_helper:
            self.write_helper.close()
        return {}


    def predict_step(self, batch, batch_idx):
        
        wav = batch["wav"]
        dur = batch["dur"]
        utt_ids = batch["utt_id"]

        xvectors = self.extract_xvectors(wav, dur)

        output = {}
        for i, utt_id in enumerate(utt_ids):
            output[utt_id] = xvectors[i]

        return output


    # @staticmethod
    # def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
    #     """
    #     Parameters you define here will be available to your model through self
    #     :param parent_parser:
    #     :param root_dir:
    #     :return:
    #     """
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)

    #     # param overwrites
    #     # parser.set_defaults(gradient_clip_val=5.0)

    #     parser.add_argument('--batch-size', default=64, type=int)
    #     parser.add_argument('--test-batch-size', default=32, type=int)
    #     parser.add_argument('--hidden-dim', default=500, type=int)
    #     parser.add_argument('--use-ecapa', default=False, action='store_true')
    #     parser.add_argument('--espnet2-model', default="", type=str)
    #     parser.add_argument('--espnet2-model-reinit', default="", type=str)
    #     parser.add_argument('--wav2vec2-model', default="", type=str)
    #     parser.add_argument('--whisper-model', default="", type=str)
    #     parser.add_argument('--pooling', default="stats", choices=['stats', 'attentive-stats', 'lde', "mha", "global-mha", "multires-mha", "ecapa-attentive-stats"])
        
    #     parser.add_argument('--pre-pooling-hidden-dim', default=-1, type=int)
    #     parser.add_argument('--pre-pooling-kernel-size', default=1, type=int)
    #     parser.add_argument('--pre-pooling-stride', default=1, type=int)
    #     # training params (opt)
        
    #     parser.add_argument("--loss", default="softmax", choices=["softmax", "am" ,"aam", "sm1", "sm2", "sm3", "mamm"])
    #     parser.add_argument("--freeze-backbone-steps", default=0, type=int)
    #     parser.add_argument("--backbone-lr-scale", default=0.01, type=float)

    #     parser.add_argument('--optimizer-name', default='adamw', type=str)
    #     parser.add_argument('--learning-rate', default=0.0005, type=float)
    #     parser.add_argument('--lr-warmup-batches', default=0, type=int)

    #     parser.add_argument('--label-smoothing', default=0.0, type=float)
    #     parser.add_argument('--entropy-regularization', default=0.0, type=float)

    #     parser.add_argument('--sample-rate', default=16000, type=int)
    #     parser.add_argument('--fbank-dim', default=40, type=int)
    #     parser.add_argument('--pooling-attention-hidden-dim', default=64, type=int)
    #     parser.add_argument('--load-pretrained-model', default=None,  type=str)
    #     parser.add_argument('--mixup-alpha', default=0.0, type=float)
        

    #     parser.add_argument('--dump-xvectors-dir', required=False, type=str)  
    #     parser.add_argument('--xvector-layer-index', default=1, type=int)
    #     parser.add_argument('--dump-predictions', required=False, type=str)        

    #     return parser
