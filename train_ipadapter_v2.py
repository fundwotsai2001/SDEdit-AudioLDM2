
import os
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
import torchaudio
from tqdm.auto import tqdm
# from transformers import SpeechT5HifiGan
from audioldm.audio import TacotronSTFT, read_wav_file
from audioldm.utils import default_audioldm_config
from scipy.io.wavfile import write
# from evaluate import LAIONCLAPEvaluator
from audioldm.audio.tools import get_mel_from_wav, _pad_spec, normalize_wav, pad_wav
def read_wav_file(filename, segment_length, augment_data=False):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = np.copy(waveform)

    waveform = pad_wav(waveform, segment_length)
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    else:
        waveform = waveform / 0.000001
    waveform = 0.5 * waveform
    return waveform
def investigate_tensor(tensor):
    if not tensor.is_leaf:
        print("The tensor is a view of another tensor.")
    else:
        print("The tensor is not a view; it is a standalone tensor.")

    if tensor.is_contiguous():
        print("The tensor is stored in a contiguous block of memory.")
    else:
        print("The tensor is not stored in a contiguous block of memory.")
def wav_to_fbank(
        filename,
        target_length=1024,
        fn_STFT=None,
        augment_data=False,
        mix_data=False,
        snr=None
    ):
    assert fn_STFT is not None
    waveform = read_wav_file(filename, target_length * 160, augment_data=augment_data)  # hop size is 160
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )
    fbank = fbank.contiguous()
    log_magnitudes_stft = log_magnitudes_stft.contiguous()
    waveform = waveform.contiguous()
    return fbank, log_magnitudes_stft, waveform



def wav_to_mel(
        original_audio_file_path,
        duration,
        augment_data=False,
        mix_data=False,
        snr=None
):
    config=default_audioldm_config()
    
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    mel, _, _ = wav_to_fbank(
        original_audio_file_path,
        target_length=int(duration * 102.4),
        fn_STFT=fn_STFT,
        augment_data=augment_data,
        mix_data=mix_data,
        snr=snr
    )
    mel = mel.unsqueeze(0)
    return mel
