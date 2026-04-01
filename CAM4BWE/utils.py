"""
@author: debes louis
"""
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import scipy
import math
matplotlib.use("Agg")  # no Tkinter

class LossPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, interval=5):
        super().__init__()
        self.save_path = save_path
        self.interval = interval
        self.train_loss = []
        self.val_loss = []
    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])

        if (epoch + 1) % self.interval == 0:
            plt.plot(figsize=(7, 5))
            plt.plot(self.train_loss, label="Train Loss")
            plt.plot(self.val_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.save_path)
            plt.close()

def callback_functions(result_foder, lr_declay_factor, patience):

    loss_plot_cb = LossPlotCallback(os.path.join(result_foder, "loss.png"), interval=5)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_declay_factor,
            patience=patience,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(result_foder, f"best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        loss_plot_cb
    ]
    return callbacks

class RunningMean:
    __slots__ = ("n", "s")
    def __init__(self):
        self.n = 0
        self.s = 0.0

    def add(self, x):
        # skip None/NaN
        if x is None:
            return
        x = float(x)
        if math.isnan(x):
            return
        self.n += 1
        self.s += x

    @property
    def mean(self):
        return self.s / self.n if self.n else float("nan")
def lowpass_filter(data, cutoff, fs=16000, order=50):
   nyquist = fs / 2
   #print(cutoff)
   normalized_cutoff = cutoff / nyquist
   sos = scipy.signal.butter(order, normalized_cutoff, btype='low', output='sos')
   signal_prefiltered = scipy.signal.sosfilt(sos, data)
   return signal_prefiltered

def load_audio(path, target_sr=16000):
    audio, fs = sf.read(path)
    if fs != target_sr:
        audio = librosa.resample(audio, fs, target_sr)
    return audio.astype(np.float32), fs

def extract_stft_mag(audio, n_fft, hop_length):
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hamming"
    )
    mag = np.abs(stft)
    mag = np.maximum(mag, 1e-8)
    phase = np.angle(stft)
    mag_db = 20 * np.log10(np.abs(mag) + 1e-8)

    return mag_db.T, phase.T  # (frames, 513)

def istft_from_mag_phase(mag_db, phase, n_fft, hop_length):

    # Convert dB -> linear magnitude
    mag = 10.0 ** (mag_db / 20.0)

    #Rebuild complex STFT
    stft = mag * np.exp(1j * phase)

    # Librosa expects (freq_bins, frames)
    stft = stft.T

    audio = librosa.istft(
        stft,
        hop_length=hop_length,
        n_fft=n_fft,
        window="hamming"
        #center=True,            # pad_mode="constant"
    )
    return audio


def mfcc_from_waveform(wave):
    n_fft = 512
    hop_length = n_fft // 2
    eps = 1e-10
    n_mfcc = 30
    n_mels = 128
    num_spec_bins = n_fft // 2 + 1
    f_min = 20.0
    f_max = 8000.0
    sample_rate = 16000

    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spec_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=f_min,
        upper_edge_hertz=f_max,
        dtype=tf.float32,
    )

    # wave: (B, T)
    wave = tf.cast(wave, tf.float32)

    stft = tf.signal.stft(
        wave,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # (B, frames, fft_bins)

    power_spec = tf.abs(stft) ** 2  # (B, frames, fft_bins)

    mel_spec = tf.tensordot(power_spec, mel_w, axes=1)  # (B, frames, n_mels)
    mel_spec = tf.maximum(mel_spec, eps)

    log_mel = tf.math.log(mel_spec)  # natural log

    mfcc_full = tf.signal.dct(log_mel, type=2, norm="ortho")  # (B, frames, n_mels)
    mfcc = mfcc_full[..., :n_mfcc]  # (B, frames, n_mfcc)
    return mfcc

def mel_from_waveform(wave):
    n_fft = 512
    hop_length = n_fft // 2
    eps = 1e-10
    n_mfcc = 30
    n_mels = 257
    num_spec_bins = n_fft // 2 + 1
    f_min = 20.0
    f_max = 8000.0
    sample_rate = 16000

    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spec_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=f_min,
        upper_edge_hertz=f_max,
        dtype=tf.float32,
    )

    # wave: (B, T)
    wave = tf.cast(wave, tf.float32)

    stft = tf.signal.stft(
        wave,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # (B, frames, fft_bins)

    power_spec = tf.abs(stft) ** 2  # (B, frames, fft_bins)
    power_spec = tf.squeeze(power_spec, axis=1)
    mel_spec = tf.tensordot(power_spec, mel_w, axes=1)  # (B, frames, n_mels)
    mel_spec = tf.maximum(mel_spec, eps)

    log_mel = tf.math.log(mel_spec)  # natural log

    return log_mel
def feature_scaler(X_wave, X_low_stft, Y_high_stft, scaler):

    WAVE, LOW_STFT, HIGH_STFT = [], [], []
    for elem in X_wave:
        WAVE.append(scaler["nb_wav"].transform(elem))
    for elem in X_low_stft:
        LOW_STFT.append(scaler["nb_stft"].transform(elem))
    for elem in Y_high_stft:
        HIGH_STFT.append(scaler["wb_stft"].transform(elem))

    return WAVE, LOW_STFT, HIGH_STFT
