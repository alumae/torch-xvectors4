import re
from typing import List, Optional, Tuple, Dict
import random
from typing import Sequence, Union
import logging
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import audiomentations
import lightning.pytorch as pl

import kaldiio
import wavemap



class WavDataset(Dataset):
    def __init__(self, datadir, extract_chunks=True, min_chunk_length=2.0, max_chunk_length=4.0,
                 label_file="utt2lang", sample_rate=16000, label2id=None,
                 noise_dir="", rir_dir="", short_noise_dir="", 
                 speed_perturbation_probability=0.0,
                 utt_max_length=0.0,
                 **kwargs):
        self.extract_chunks = extract_chunks
        self.min_length = min_chunk_length
        self.max_length = max_chunk_length
        self.sample_rate = sample_rate
        self.utt2label = {}
        self.speed_perturbation_probability = speed_perturbation_probability
        self.utt_max_length = utt_max_length

        for l in open(f"{datadir}/{label_file}"):
            ss = l.split()
            self.utt2label[ss[0]] = ss[1]

        self.labels = list(sorted(set(self.utt2label.values())))
        if label2id is None:
            self.label2id = {label: i for i, label in enumerate(self.labels)}
        else:
            self.label2id = label2id
            for label in self.labels:
                assert label in label2id

        logging.info(f"Reading wav locations from {datadir}/wav.scp")
        self.utt2file = {}
        self.utts = []
        self.utt2index = {}
        for line in open(f"{datadir}/wav.scp"):
            wav_id, location = line.split(maxsplit=1)
            if wav_id in self.utt2label:
                self.utt2file[wav_id] = location.strip() 
                self.utt2index[wav_id] = len(self.utts)
                self.utts.append(wav_id)

        self.num_labels = len(self.labels)
        self.utt2dur = {}
        for l in open(f"{datadir}/utt2dur"):
            ss = l.split()
            if ss[0] in self.utt2label:
                self.utt2dur[ss[0]] = float(ss[1])
        self.total_dur = sum(self.utt2dur.values())

        self.augment = None
        augmentations = []
        if rir_dir != "":
            augmentations.append(audiomentations.ApplyImpulseResponse(ir_path=rir_dir, p=0.5, lru_cache_size=1024, leave_length_unchanged=True))
        if noise_dir != "":
            augmentations.append(audiomentations.AddBackgroundNoise(sounds_path=noise_dir, p=0.5, lru_cache_size=1024))
        augmentations.append(audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3))
        if short_noise_dir != "":
            augmentations.append(audiomentations.AddShortNoises(sounds_path=short_noise_dir, p=0.5, lru_cache_size=1024))
        if speed_perturbation_probability > 0.0:
            augmentations.append(audiomentations.Resample(min_sample_rate=sample_rate * 0.9, max_sample_rate=sample_rate * 1.1, p=speed_perturbation_probability))

        if len(augmentations) > 0:
            self.augment = audiomentations.Compose(augmentations)


    def get_wav_audio(self, audio_path):
        """Load the wav file at the given file path. Only 16-bit wavs are supported."""

        if audio_path.endswith("|"):
            audio_path = audio_path.replace("wav-copy", "wav-copy --print-args=false")
            utt_sample_rate, sound_np = kaldiio.load_mat(audio_path)
            assert(utt_sample_rate == self.sample_rate)
            #wav_tensor = torch.FloatTensor(wav_np_array / 2**15)
        else:
            sound_np = wavemap(audio_path, 'r')
            assert(sound_np.sample_rate == self.sample_rate)
        assert(sound_np.dtype == np.int16)
        # FIXME: resample if necessary        

        assert(len(sound_np.shape) == 1)
        return sound_np.astype(np.float32) / 32768.0

    def __getitem__(self, index):
        utt = self.utts[index]
        return {"index": index,
                "utt": utt, 
                "wav": self.utt2file[utt],
                "label": self.utt2label[utt],
                "label_id": self.label2id[self.utt2label[utt]],
                "dur": self.utt2dur[utt]}

    def __len__(self):
        return len(self.utts)

    def index2utt(self, index):
        return self.utts[index]

    def utt2index(self, utt):
        return self.utt2index[utt]



    def collater(self, samples):
        """Merge a list of wavs to form a mini-batch.

        Args:
            samples (List[dict]): wavs to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}

        indexes = [s["index"] for s in samples]
        wavs = [s["wav"] for s in samples]
        utts = [s["utt"] for s in samples]

        audios = []
        durs = []

        if self.extract_chunks:
            chunk_length = random.uniform(self.min_length, self.max_length)
            chunk_length_in_samples = int(chunk_length * self.sample_rate)
            collated_audio = torch.zeros(len(wavs), chunk_length_in_samples)

            start_positions = []
            for i, wav in enumerate(wavs):
                audio_tensor = self.get_wav_audio(wav)
                if self.augment is not None:
                    audio_tensor = self.augment(samples=audio_tensor, sample_rate=self.sample_rate)

                start_pos = random.randint(0, max(0, len(audio_tensor) - chunk_length_in_samples))
                start_positions.append(int(start_pos * 100))
                current_chunk_length = min(chunk_length_in_samples, len(audio_tensor))
                chunk = audio_tensor[start_pos:start_pos+current_chunk_length]

                audios.append(torch.from_numpy(chunk))
                durs.append(len(chunk))
        else:
            start_positions = []
            for i, wav in enumerate(wavs):
                audio_tensor = self.get_wav_audio(wav)
                if self.utt_max_length > 0.0:
                    audio_tensor = audio_tensor[:int(self.utt_max_length * self.sample_rate)]
                start_positions.append(0)
                if self.augment is not None:
                    audio_tensor = self.augment(samples=audio_tensor, sample_rate=self.sample_rate)
                audios.append(torch.from_numpy(audio_tensor))
                durs.append(len(audio_tensor))

        batch = {
            "index": torch.tensor(indexes),
            "wav": torch.nn.utils.rnn.pad_sequence(audios, batch_first=True),
            "dur": torch.tensor(durs),
            "label_id": torch.tensor([s["label_id"] for s in samples]),
            "utt_id": utts
        }

        return batch


class DataModule(pl.LightningDataModule):
    def __init__(self, 
        train_dir: Union[None, str],
        dev_dir, #;Union[None, str, List[str]],
        predict_dir: Union[None, str],
        min_chunk_length: float = 2.0, 
        max_chunk_length: float = 4.0,
        noise_dir: str = "", 
        rir_dir: str = "", 
        short_noise_dir = "",
        speed_perturbation_probability: float = 0.0, 
        label_file: str ="utt2lang",
        sample_rate: int = 16000,
        batch_size: int = 16,
        num_workers: int = 4,
        label2id: Dict = None
        
    ):
        super().__init__()
        self.save_hyperparameters()
        if isinstance(self.hparams.dev_dir, str):
            self.hparams.dev_dir = [self.hparams.dev_dir]
        
        self.label2id = label2id    


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = WavDataset(
                datadir=self.hparams.train_dir,
                noise_dir=self.hparams.noise_dir, 
                short_noise_dir=self.hparams.short_noise_dir,
                rir_dir=self.hparams.rir_dir,
                speed_perturbation_probability=self.hparams.speed_perturbation_probability,
                extract_chunks=True,                
                min_chunk_length=self.hparams.min_chunk_length,
                max_chunk_length=self.hparams.max_chunk_length,
                label_file=self.hparams.label_file,
                label2id=self.label2id
            )
            if self.label2id == None:
                assert self.train_dataset.label2id != None
                self.label2id = self.train_dataset.label2id
                
        

        if stage == "fit" or stage == "validate": 
            self.val_datasets = [
                WavDataset(
                    datadir=dir,
                    extract_chunks=False,   
                    utt_max_length=30.0,             
                    label_file=self.hparams.label_file,
                    label2id=self.label2id
                ) for dir in self.hparams.dev_dir]


        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            raise Error("Not implemented")

        if stage == "predict":
            self.predict_dataset = WavDataset(
                    datadir=self.hparams.predict_dir,
                    extract_chunks=False,   
                    utt_max_length=30.0,             
                    label_file=self.hparams.label_file,
                    label2id=self.label2id)
                


    def train_dataloader(self):
        return DataLoader(
           self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collater,
        )

    def val_dataloader(self):
        return [DataLoader(
           val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=val_dataset.collater,
        ) for val_dataset in self.val_datasets]

    def test_dataloader(self):
        raise Error("Not implemented")

    def predict_dataloader(self):
        return DataLoader(
           self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.predict_dataset.collater,
        )
