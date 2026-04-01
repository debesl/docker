"""
@author: debes louis
"""
import tensorflow as tf
import os
from enum import Enum
import numpy as np
import soundfile as sf
from DataGenerator import generate_dataset_testing
from utils import istft_from_mag_phase
from evaluate import evaluate_dataset
import joblib
from model import NeuronalNetworkType


#TODO: debugging scaler
import utils
import DataGenerator
from sklearn.preprocessing import StandardScaler
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_physical_devices('GPU')
        print(len(gpus), 'Physical GPUs', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

class ProcedureMode(Enum):
    All = 1,
    Predict = 2,
    Evaluating = 3
def extend_phase_by_mirroring(phase_low, full_bins):
    """
    Extend low-band phase (0–4 kHz) to full band (0–8 kHz)
    by mirroring.

    Args:
        phase_low: (frames, low_bins)
        full_bins: int

    Returns:
        phase_full: (frames, full_bins)
    """
    frames, low_bins = phase_low.shape
    high_bins = full_bins - low_bins

    # Mirror low-band phase
    phase_high = phase_low[:, ::-1]

    # Tile if needed
    if phase_high.shape[1] < high_bins:
        repeats = int(np.ceil(high_bins / phase_high.shape[1]))
        phase_high = np.tile(phase_high, (1, repeats))

    phase_high = phase_high[:, :high_bins]

    phase_full = np.concatenate([phase_low, phase_high], axis=1)

    return phase_full

def combine_nb_pred(pred_mag, nb_mag, full_bins):
    frames, low_bins = nb_mag.shape
    # pred_mag should be (frames, full_bins)
    assert pred_mag.shape[0] == frames
    assert pred_mag.shape[1] >= full_bins

    predicted_mag_high = pred_mag[:, low_bins:full_bins]
    audio = np.concatenate([nb_mag, predicted_mag_high], axis=1)

    return audio

def fold_phase_only(phase):
    """
    bins_axis : int
        Which axis is the frequency-bin axis. For (frames, bins) use -1 or 1.
        For (bins, frames) use 0.
    """
    # Build upper band by mirroring lower band
    phase_wo_nyquist = phase[:, :-1]
    upper = phase_wo_nyquist[..., ::-1]
    #zeros = np.zeros((phase.shape[0], 1), dtype=phase.dtype)
    # Concatenate: (low | folded upper)
    full = np.concatenate([phase, upper], axis=-1)  # (..., 256)

    return full

def reconstruct(predicted_mag_wb, narrowband_mag, scaler, hop_length, n_fft, phase_nb, pad):
    bins = predicted_mag_wb.shape[-1]
    # only take bottom half of fft spektrum upper half is 0 anyways
    nb_cut = (phase_nb.shape[1] // 2) + 1

    phase_nb = phase_nb[:, :nb_cut]
    narrowband_mag = narrowband_mag[:, :nb_cut]

    # phase_full = extend_phase_by_mirroring(phase_low=phase_nb, full_bins=bins)
    phase_full = fold_phase_only(phase_nb)

    combined_mag = combine_nb_pred(pred_mag=predicted_mag_wb, nb_mag=narrowband_mag, full_bins=bins)

    combined_mag = scaler.inverse_transform(combined_mag)

    audio = istft_from_mag_phase(mag_db=combined_mag, phase=phase_full, n_fft=n_fft, hop_length=hop_length)

    if pad > 0:
        audio = audio[:-pad]

    return audio

def save_audio(reconsturcted_audio, sr, path, filename):

    if not os.path.exists(os.path.join(path, "predict")):
        os.mkdir(os.path.join(path, "predict"))

    out_path = os.path.join(path, "predict",  filename)
    sf.write(out_path, reconsturcted_audio.astype(np.float32), int(sr))

def save_to_txt(filepath, results):
    path = os.path.join(filepath, "metrics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PESQ: {results[0]}\n")
        f.write(f"visqol_score: {results[1]}\n")
        f.write(f"lsd: {results[2]}\n")
        f.write(f"lsd_high: {results[3]}\n")
        f.write(f"vgg_dist: {results[4]}\n")

if __name__ == '__main__':

    # ============================================================================
    #  Data preProcessing
    # ============================================================================
    # Signal / STFT parameters
    # ========================
    sr = 16000
    n_fft = 512
    hop_length = 256
    waveform_len = n_fft  # temporal context (NOT equal to n_fft)
    num_frames=50
    # ============================================================================
    #  Trianing Prameters
    # ============================================================================
    model_type = NeuronalNetworkType.mel_mlp_x_attetnion

    data_root = r"F:\Feature\TIMIT_CAM4BWE\test"
    model_path = r"F:\results\CAM4BWE\mel_mlp_x_attetnion_3dense_out_epochs_50_bs_64_lr_0.0001_nfft_512_hop_256_decay_2e-05_patience_5"
    mode = ProcedureMode.All

    model = tf.keras.models.load_model(os.path.join(model_path, "best_model.h5"))
    scaler_wb = joblib.load(f"scaler/feature_scaler_wb_mag_nfft_{n_fft}.bin")
    scaler_nb = joblib.load(f"scaler/feature_scaler_nb_mag_nfft_{n_fft}.bin")
    scaler_wav = joblib.load(f"scaler/feature_scaler_nb_wav_nfft_{n_fft}.bin")
    scaler = {"wb": scaler_wb, "nb": scaler_nb, "wav": scaler_wav}
    X_wave, X_low_stft, Y_high_stft, phase, fn, pad = generate_dataset_testing(data_root,
                                                                                 scaler,
                                                                                 hop_length,
                                                                                 n_fft,
                                                                                 waveform_len,
                                                                                 sr,
                                                                                 num_frames=num_frames,
                                                                                 cut_off=4000,
                                                                                 number_of_files=10000)

    if mode == mode.Predict or mode == mode.All:
        print(f"Predict on Dataset: {data_root}")
        for idx, _ in enumerate(X_wave):
            wav_test = X_wave[idx]
            stft_test = X_low_stft[idx]
            if model_type == NeuronalNetworkType.mlp_stft:
                pred_wb_stft = model.predict(stft_test)
            elif model_type == NeuronalNetworkType.mel_mlp_x_attetnion:
                mel = utils.mel_from_waveform(wav_test)
                pred_wb_stft = model.predict([mel, stft_test])
            else:
                pred_wb_stft = model.predict([wav_test, stft_test])

            reconsturcted_audio = reconstruct(predicted_mag_wb=pred_wb_stft, narrowband_mag=X_low_stft[idx],
                                              scaler=scaler["wb"], hop_length=hop_length,
                                              n_fft=n_fft, phase_nb=phase[idx], pad=pad[idx])

            save_audio(reconsturcted_audio, sr,  path=model_path, filename=fn[idx])

    if mode == mode.Evaluating or mode == mode.All:
        print(f"Evaluate Dataset: {data_root}")
        ref_dir = os.path.join(data_root, "clean")
        predict_dir = os.path.join(model_path, "predict")
        results = evaluate_dataset(ref_dir, predict_dir, sr=sr, n_fft=n_fft, hop=hop_length)
        save_to_txt(model_path, results)
