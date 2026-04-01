import os
import numpy as np
import utils
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def extract_pairs_padding(lowband, mag_low, waveform_len, hop_length):

    half = waveform_len // 2
    n_frames = len(mag_low)

    # Pad both waveforms so start/end never go out of bounds
    # 'reflect' tends to work better than zeros for audio edges, but 'constant' is also ok. regarding to stackoverflow??
    extra_right = max(0, (n_frames - 1) * hop_length + waveform_len - len(lowband))
    pad_left = half
    pad_right = extra_right
    mode = "constant"

    low_pad = np.pad(lowband, (pad_left, pad_right), mode=mode, constant_values=(0, 0))
    X_wave = np.empty((n_frames, waveform_len), dtype=lowband.dtype)

    for t in range(n_frames):
        center = t * hop_length
        start = center
        end = center + waveform_len
        # because we padded by `half` on the left, shift indices by +half
        start += half
        end += half
        X_wave[t] = low_pad[start:end]  # x1

    return np.asarray(X_wave)

def create_dataset_stft(file_path, hop_length, n_fft, waveform_len, cut_off, sr):

   audio_signal, fs = utils.load_audio(file_path)
   assert fs == sr, f"Expected sampling rate to be {sr}, but audio has {fs}"
   frame_length = n_fft

   pad_width = (frame_length - len(audio_signal) % frame_length) % frame_length
   audio_signal = np.pad(audio_signal, (0, pad_width), mode='constant')

   high_stft_mag, high_stft_phase = utils.extract_stft_mag(audio_signal.astype(np.float32),
                                                           n_fft=n_fft,
                                                           hop_length=hop_length)

   low_audio = utils.lowpass_filter(audio_signal, cutoff=cut_off, fs=fs)
   low_stft_mag, low_stft_phase = utils.extract_stft_mag(low_audio.astype(np.float32),
                                                         n_fft=n_fft,
                                                         hop_length=hop_length)

   wav = extract_pairs_padding(low_audio, low_stft_mag, waveform_len, hop_length)

   return high_stft_mag, low_stft_mag, wav, low_stft_phase, pad_width


def generate_dataset_training(target_folder, hop_length, n_fft, waveform_len, sr, cut_off, number_of_files):
    High_STFT_FEATURES, Low_STFT_FEATURES, Low_WAV_FEATURES = [], [], []

    processed = 0
    for root, dirs, files in os.walk(target_folder):
       for file in files:
           if file.endswith('.wav'):
               file_pathlow = os.path.join(root, file)

               high, low, wav, _, _ = create_dataset_stft(file_pathlow, hop_length, n_fft, waveform_len, cut_off, sr)
               High_STFT_FEATURES.extend(high)
               Low_STFT_FEATURES.extend(low)
               Low_WAV_FEATURES.extend(wav)
               processed += 1
               if processed >= number_of_files:
                   break  # breaks inner loop
           if processed >= number_of_files:
               break  # breaks outer loop

    Low_STFT_FEATURES = np.array(Low_STFT_FEATURES)
    High_STFT_FEATURES = np.array(High_STFT_FEATURES)
    Low_WAV_FEATURES = np.array(Low_WAV_FEATURES)

    stft_low_scaler = StandardScaler()
    wav_low_scaler = StandardScaler()
    stft_high_scaler = StandardScaler()

    stft_low_scaler.fit(Low_STFT_FEATURES)
    wav_low_scaler.fit(Low_WAV_FEATURES)
    stft_high_scaler.fit(High_STFT_FEATURES)

    X_low_stft = stft_low_scaler.transform(Low_STFT_FEATURES)
    X_wave = wav_low_scaler.transform(Low_WAV_FEATURES)
    Y_high_stft = stft_high_scaler.transform(High_STFT_FEATURES)

    if not os.path.exists("./scaler"):
        os.makedirs("./scaler")
    scaler_fn = "./scaler/feature_scaler"
    joblib.dump(stft_high_scaler, f"{scaler_fn}_wb_mag_nfft_{n_fft}.bin", compress=False)
    joblib.dump(stft_low_scaler, f"{scaler_fn}_nb_mag_nfft_{n_fft}.bin", compress=False)
    joblib.dump(wav_low_scaler, f"{scaler_fn}_nb_wav_nfft_{waveform_len}.bin", compress=False)


    return X_wave, X_low_stft, Y_high_stft

def generate_dataset_testing(target_folder,scaler, hop_length, n_fft, waveform_len, sr,  cut_off, number_of_files):
    X_wave, X_low_stft, Y_high_stft, phase, pad = [], [], [], [], []
    fn = []

    processed = 0
    for root, dirs, files in os.walk(target_folder):
       for file in files:
           if file.endswith('.wav'):
               file_pathlow = os.path.join(root, file)

               high, low, wav, nb_phase, padding = create_dataset_stft(file_pathlow, hop_length, n_fft, waveform_len, cut_off, sr)

               X_low_stft_scaled = scaler["nb"].transform(low)
               X_wave_scaled = scaler["wav"].transform(wav)
               Y_high_stft_scaled = scaler["wb"].transform(high)

               fn.append(file)
               pad.append(padding)
               Y_high_stft.append(Y_high_stft_scaled)
               X_low_stft.append(X_low_stft_scaled)
               X_wave.append(X_wave_scaled)
               phase.append(nb_phase)

               processed += 1
               if processed >= number_of_files:
                   break  # breaks inner loop
           if processed >= number_of_files:
               break  # breaks outer loop


    return X_wave, X_low_stft, Y_high_stft, phase, fn, pad

def build_frames(X, num_frames):
    T, F = X.shape
    B = (T + num_frames - 1) // num_frames  # ceil(T / num_frames)
    target_T = B * num_frames
    pad_T = target_T - T

    if pad_T > 0:
        X = np.pad(X, ((0, pad_T), (0, 0)), mode="constant", constant_values=0)

    X = X.reshape(B, num_frames, F)

    return X