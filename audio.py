"""Wrapper class for audio clips."""

import numpy as np
import librosa
import soundfile as sf


class Audio:
    """Wrapper for audio clips."""
    
    def __init__(self, samples: np.ndarray, rate: int):
        if samples.ndim == 1:
            samples = np.expand_dims(samples, axis=1)
        elif samples.ndim > 2:
            raise ValueError(
                f'The shape {samples.shape} is of too high dimension.')

        if samples.shape[0] < samples.shape[1]:
            samples = samples.T

        self._samples = samples
        self._rate = rate

    @classmethod
    def _new_audio(cls, samples: np.ndarray, rate: int) -> 'Audio':
        return cls(samples, rate)

    @property
    def samples(self) -> np.ndarray:
        return self._samples
    
    @property
    def rate(self) -> int:
        return self._rate
    
    @property
    def channels(self) -> int:
        return self._samples.shape[1]
    
    def __len__(self) -> int:
        return self._samples.shape[0]
    
    @property
    def duration(self) -> float:
        return len(self._samples) / self._rate

    # Change to module-level function instead? `read_wav` too.
    @classmethod
    def read_flac(cls, path: str, dtype: str = 'float64') -> 'Audio':
        with open(path, 'rb') as f:
            samples, rate = sf.read(f, dtype=dtype)
        return cls(samples, rate)

    @classmethod
    def read_wav(cls, path: str, dtype: str = 'float64') -> 'Audio':
        samples, rate = sf.read(path, dtype=dtype)
        return cls(samples, rate)

    def write_wav(self, path: str, dtype: str = 'float64'):
        sf.write(path, self._samples[:, 0], self._rate)

    def resample(self, target_sr: int) -> 'Audio':
        samples = librosa.resample(
            y=np.squeeze(self._samples),
            orig_sr=self._rate,
            target_sr=target_sr,
            res_type='scipy',  # Bug with 44.1 kHz to 16 kHz using soxr_hq...
        )
        return self._new_audio(samples, target_sr)
    
    def repetitive_crop(self, length: int) -> 'Audio':
        new_samples = self._samples
        while new_samples.shape[0] < length:
            new_samples = np.concatenate(
                (new_samples, new_samples), axis=0)
        new_samples = new_samples[:length, :]
        return self._new_audio(new_samples, self._rate)

    def peak_normalize(self, max_amplitude: float = 0.95) -> 'Audio':
        max_val = max(np.max(self.samples), 1e-8)
        return self._new_audio(self.samples * max_amplitude / max_val, self._rate)

    def scale(self, gain: float) -> 'Audio':
        return self._new_audio(self.samples * gain, self._rate)
