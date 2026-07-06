# backend/utils/audio.py
import io
from pathlib import Path
from typing import Union
import soundfile as sf
import torch
import torchaudio

def load_audio(device, audio_path: Union[str, Path], sample_rate: int):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise RuntimeError(f"Audio file not found: {audio_path}")

    audio, file_sample_rate = torchaudio.load(audio_path)
    
    audio = audio.to(device)

    # if mono, convert to stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    if file_sample_rate != sample_rate:
        resample = torchaudio.transforms.Resample(file_sample_rate, sample_rate).to(device)
        audio = resample(audio)

    return audio


def save_audio_to_buffer(tensor: torch.Tensor, sample_rate: int, format: str = "wav") -> io.BytesIO:
    """Saves a torch tensor [channels, time] to a BytesIO buffer using soundfile."""
    buffer = io.BytesIO()
    # Move to CPU, detach, and convert to numpy
    data = tensor.detach().cpu().numpy()
    # Soundfile expects shape [time, channels]
    if data.ndim == 2:
        data = data.T
    sf.write(buffer, data, sample_rate, format=format.upper())
    buffer.seek(0)
    return buffer


def save_audio(tensor: torch.Tensor, filepath: Union[str, Path], sample_rate: int, format: str = "wav") -> None:
    """Saves a torch tensor [channels, time] directly to a file path using soundfile."""
    # Move to CPU, detach, and convert to numpy
    data = tensor.detach().cpu().numpy()
    # Soundfile expects shape [time, channels]
    if data.ndim == 2:
        data = data.T
    with open(filepath, 'wb') as f:
        sf.write(f, data, sample_rate, format=format.upper())

