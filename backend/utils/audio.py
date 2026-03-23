# backend/utils/audio.py
import os
import torchaudio

def load_audio(device, audio_path: str, sample_rate):
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")

    audio, file_sample_rate = torchaudio.load(audio_path)

    # if mono, convert to stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    if file_sample_rate != sample_rate:
        resample = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        audio = resample(audio)

    return audio.to(device)
