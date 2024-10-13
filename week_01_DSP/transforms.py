from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        windows = []
        waveform = np.pad(waveform, ((self.window_size)//2,(self.window_size)//2))
        for i in range(0,len(waveform)-self.window_size+1,self.hop_length):
            windows.append(waveform[i:i+self.window_size])

        return np.array(windows)
    

class Hann:
    def __init__(self, window_size=1024):
        # Your code here
        self.hann = scipy.signal.windows.hann(window_size,sym=False)
        # ^^^^^^^^^^^^^^

    
    def __call__(self, windows):
        # Your code here
        return self.hann*windows
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        # Your code here
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^

        dft = np.fft.rfft(windows)
        if self.n_freqs is not None:
            dft = dft[:,:self.n_freqs]
        spec = np.abs(dft)
        return spec


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        # Your code here
        self.mel = librosa.filters.mel(sr = sample_rate,n_fft = n_fft,n_mels = n_mels)
        self.inverse_mel = np.linalg.pinv(self.mel)
        # ^^^^^^^^^^^^^^


    def __call__(self, spec):
        # Your code here
        mel = spec@self.mel.T
        # ^^^^^^^^^^^^^^
        
        return mel

    def restore(self, mel):
        # Your code here
        spec = mel@self.inverse_mel.T
        

        # ^^^^^^^^^^^^^^

        return spec


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        # return self.square(self.fft(self.hann(self.windowing(waveform))))
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class PitchUp:
    def __init__(self, num_mels_up):
        # Your code here
        self.num_mels_up = num_mels_up
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        pad = np.zeros(mel[:,:self.num_mels_up].shape)
        mel = np.hstack((pad,mel[:,:-self.num_mels_up]))
        return mel
        # Your code here
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class PitchDown:
    def __init__(self, num_mels_down):
        # Your code here

        self.num_mels_up = num_mels_down
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):

        pad = np.zeros(mel[:,:self.num_mels_up].shape)
        mel = np.hstack((mel[:,self.num_mels_up:],pad))
        # Your code here
        return mel
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


import torch
class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        # Your code here
        self.speed_up_factor = speed_up_factor
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        return np.array(torch.adaptive_avg_pool1d(torch.Tensor(mel.T),int(mel.shape[0]*self.speed_up_factor))).T
        # ^^^^^^^^^^^^^^



class Loudness:
    def __init__(self, loudness_factor):
        self.loudness_factor = loudness_factor
        # Your code here
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        return mel*self.loudness_factor
        # Your code here
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class TimeReverse:
    def __call__(self, mel):
        # Your code here
        return mel[::-1,:]
        # ^^^^^^^^^^^^^^



class VerticalSwap:
    def __call__(self, mel):
        # Your code here
        return mel[:,::-1]
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        # Your code here
        self.quantile = quantile
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        mel = mel.copy()
        perc = np.percentile(mel,self.quantile*100)
        # Your code here
        for i in range(mel.shape[0]):
            idc = mel[i][np.where(mel[i] <perc)]
            mel[i][np.where(mel[i] <perc)] = 0


        return mel
        # ^^^^^^^^^^^^^^



class Cringe1:
    def __init__(self):
        # Your code here
        pass
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):

        mel1=mel.copy()
        mel1[0::2] = mel[::-2]

        return mel1
        # Your code here
        # raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class Cringe2:
    def __init__(self):
        # Your code here
        # ^^^^^^^^^^^^^^
        pass


    def __call__(self, mel):
        # Your code here
        mel1=mel.copy()
        mel1[1::2] = mel[::-2][1:]

        return mel1