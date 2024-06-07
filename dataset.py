"""Dataset loader for VCC2018, BVCC, and NISQA."""

import os
import sys
from typing import Callable, Optional, Sequence, Union

import gin
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import tqdm

import audio
import utils


@gin.configurable
class Vcc2018(Dataset):
    """The VCC2018 dataset."""
    
    def __init__(
        self,
        data_path: str = '../../datasets/VCC2018/testVCC2',
        valid: str = 'train',
        sample_rate: int = 16000,
    ):
        """Initializes the instance.
        
        Args:
            data_path: Path to the dataset.
            valid: The data type. Can be 'train', 'val', or 'test'.
            sample_rate: The target sample rate of the data.
        """
        self._data_path = data_path
        if valid == 'train':
            self._df = pd.read_csv(os.path.join(data_path, 'training_data.csv'))
        elif valid == 'val':
            self._df = pd.read_csv(os.path.join(data_path, 'valid_data.csv'))
        elif valid == 'test':
            self._df = pd.read_csv(os.path.join(data_path, 'testing_data.csv'))
        else:
            raise ValueError(f'{valid=} is not valid.')
        self._df = self._df[['WAV_PATH', 'MEAN']].groupby(by='WAV_PATH').mean().reset_index()
        self._num_samples = len(self._df)
        self._valid = valid
        self._sample_rate = sample_rate

        self._labels = self._load_labels()
        self._mag_specs = self._load_clips()
  
    @property
    def label_type(self) -> str:
        """The type of the label."""
        return 'mos'

    def _load_labels(self) -> list[float]:
        """Loads the labels."""
        return self._df['MEAN'].to_list()

    def _load_clips(self) -> list[np.ndarray]:
        """Loads the clips and transforms to spectrograms"""
        mag_specs = []
        for path in tqdm.tqdm(
            self._df['WAV_PATH'], total=self._num_samples, desc='Loading clips...',
        ):
            wav, sr = librosa.load(os.path.join(self._data_path, path), sr=self._sample_rate)
            signal = audio.Audio(wav, sr)
            signal = signal.repetitive_crop(10 * self._sample_rate)
            samples = np.squeeze(signal.samples)
            spec = utils.stft(samples)
            mag_specs.append(spec)

        return mag_specs
   
    def __getitem__(self, idx: int) -> tuple[np.ndarray, float]:
        """Returns a spectrogram and label thereof."""
        return self._mag_specs[idx], self._labels[idx]
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._mag_specs)
 
    def collate_fn(self, batch: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a batch consisting of tensors."""
        mag_specs, labels = zip(*batch)
        mag_specs = torch.FloatTensor(np.array(mag_specs))
        labels = torch.FloatTensor(labels)
        return mag_specs, labels


@gin.configurable
class Bvcc(Dataset):
    """The BVCC dataset."""
    
    def __init__(
        self,
        data_path: str = '../../datasets/BVCC/DATA',
        valid: str = 'train',
        sample_rate: int = 16000,
    ):
        """Initializes the instance.
        
        Args:
            data_path: Path to the dataset.
            valid: The data type. Can be 'train', 'val', or 'test'.
            sample_rate: The target sample rate of the data.
        """
        self._data_path = data_path
        self._df = self._load_df(valid)
        self._num_samples = len(self._df)
        self._valid = valid
        self._sample_rate = sample_rate

        self._labels = self._load_labels()
        self._mag_specs = self._load_clips()
    
    def _load_df(self, valid: str) -> pd.DataFrame:
        names = {'train': 'TRAINSET', 'val': 'DEVSET', 'test': 'TESTSET'}
        if valid not in names:
            raise ValueError(f'{valid=} is not valid.')
        # read metadata
        filenames = []
        individual_scores = {}
        with open(os.path.join(self._data_path, 'sets', names[valid]), 'r') as f:
            lines = f.read().splitlines()

            # line has format <system, wav_name, score, _, judge_name>
            for line in lines:
                if line:
                    _, filename, score, _, rater_name = line.split(",")
                    if filename in individual_scores:
                        individual_scores[filename].append(int(score))
                        continue
                    else:
                        individual_scores[filename] = [int(score)]
                    filenames.append(filename)

        return pd.DataFrame({
            'filenames': filenames,
            'mos': [np.mean(scores) for scores in individual_scores.values()]
        })

    @property
    def label_type(self) -> str:
        """The type of the label."""
        return 'mos'

    def _load_labels(self) -> list[float]:
        """Loads the labels."""
        return self._df['mos'].to_list()

    def _load_clips(self) -> list[np.ndarray]:
        """Loads the clips and transforms to spectrograms"""
        mag_specs = []
        for path in tqdm.tqdm(
            self._df['filenames'], total=self._num_samples, desc='Loading clips...',
        ):
            wav, sr = librosa.load(os.path.join(self._data_path, 'wav', path), sr=self._sample_rate)
            signal = audio.Audio(wav, sr)
            signal = signal.repetitive_crop(10 * self._sample_rate)
            samples = np.squeeze(signal.samples)
            spec = utils.stft(samples)
            mag_specs.append(spec)

        return mag_specs
   
    def __getitem__(self, idx: int) -> tuple[np.ndarray, float]:
        """Returns a spectrogram and label thereof."""
        return self._mag_specs[idx], self._labels[idx]
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._mag_specs)
 
    def collate_fn(self, batch: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a batch consisting of tensors."""
        mag_specs, labels = zip(*batch)
        mag_specs = torch.FloatTensor(np.array(mag_specs))
        labels = torch.FloatTensor(labels)
        return mag_specs, labels

    
@gin.configurable
class Nisqa(Dataset):
    """The NISQA dataset."""
    
    def __init__(
        self,
        data_path: str = '../../datasets/NISQA_Corpus',
        valid: str = 'train',
        label_type: str = 'mos',
        sample_rate: int = 16000,
    ):
        """Initializes the instance.
        
        Args:
            data_path: Path to the dataset.
            valid: The data type. Can be 'train', 'val', or 'test'.
            label_type: The type of label. Can be 'mos', 'noi', 'col', 'dis', or 'loud'.
            sample_rate: The sample rate of the data. Note that the data is
                stored at 48 kHz.
        """
        self._data_path = data_path
        if valid == 'train':
            self._df = pd.read_csv(os.path.join(data_path, 'NISQA_TRAIN_SIM', 'NISQA_TRAIN_SIM_file.csv'))
        elif valid == 'val' or valid == 'test':
            self._df = pd.read_csv(os.path.join(data_path, 'NISQA_VAL_SIM', 'NISQA_VAL_SIM_file.csv'))
            self._df = self._df.sample(frac=1, random_state=1997)
            if valid == 'val':
                self._df = self._df.iloc[:1250]
            else:
                self._df = self._df.iloc[1250:]
        else:
            raise ValueError(f'{valid=} is not valid.')
        self._num_samples = len(self._df)
        self._valid = valid
        self._sample_rate = sample_rate
        self._label_type = label_type

        self._labels = self._load_labels()
        self._mag_specs = self._load_clips()

    @property
    def label_type(self) -> str:
        """Returns the type of the label (i.e., 'mos', 'noi', 'col', 'dis', or 'loud')."""
        return self._label_type

    def _load_labels(self) -> list[float]:
        """Loads the labels."""
        return self._df[self._label_type].to_list()

    def _load_clips(self) -> list[np.ndarray]:
        """Loads the clips and transforms to spectrograms"""
        mag_specs = []
        for path in tqdm.tqdm(
            self._df['filepath_deg'], total=self._num_samples, desc='Loading clips...',
        ):
            wav, _ = librosa.load(os.path.join(self._data_path, path), sr=self._sample_rate)
            signal = audio.Audio(wav, self._sample_rate)
            signal = signal.repetitive_crop(10 * self._sample_rate)
            samples = np.squeeze(signal.samples)
            spec = utils.stft(samples)
            mag_specs.append(spec)

        return mag_specs
   
    def __getitem__(self, idx: int) -> tuple[np.ndarray, float]:
        """Returns a spectrogram and label thereof."""
        return self._mag_specs[idx], self._labels[idx]
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._mag_specs)
 
    def collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a batch consisting of tensors."""
        mag_specs, labels = zip(*batch)
        mag_specs = torch.FloatTensor(np.array(mag_specs))
        labels = torch.FloatTensor(labels)
        return mag_specs, labels
    

@gin.configurable
def get_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Returns a dataloader of the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
