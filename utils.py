from typing import Optional

import gin
import numpy as np
import librosa


@gin.configurable
def stft(
    samples: np.ndarray,
    win_length: int = 320,
    hop_length: int = 160,
    n_fft: int = 320,
    use_log: bool = True,
    use_magnitude: bool = True,
    n_mels: Optional[int] = None,
) -> np.ndarray:
    if use_log and not use_magnitude:
        raise ValueError('Log is only available if the magnitude is to be computed.')
    if n_mels is None:
        spec = librosa.stft(y=samples, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
    else:
        spec = librosa.feature.melspectrogram(
            y=samples, win_length=win_length, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels
        )
    spec = spec.T
    if use_magnitude:
        spec = np.abs(spec)
    if use_log:
        spec = np.clip(spec, 10 ** (-7), 10 ** 7)
        spec = np.log10(spec)
    return spec
    
