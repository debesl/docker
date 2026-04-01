"""
@author: debes louis
"""
import os
from natsort import natsorted
import re
import subprocess
import pathlib
import numpy as np
import librosa
import tensorflow as tf
from pesq import pesq
from tqdm import tqdm
import soundfile as sf
import tensorflow_hub as hub

import LSD.lsd
from utils import RunningMean
from VISQOL.matlab_wrapper import run_visqol_matlab_cli
# ----------------------------
# 1) LSD: Log Spectral Distance
# ----------------------------

# ----------------------------
# 2) VGG distance using VGGish embeddings
# ----------------------------
class VGGishDistance:
    """
    Uses TF-Hub VGGish model to compute embeddings and returns mean L2 distance.
    The hub model interface can vary; we try common calling patterns.
    """
    def __init__(self, hub_handle: str = "https://tfhub.dev/google/vggish/1", sr: int = 16000):
        self.sr = sr
        self.model = hub.load(hub_handle)

    def _embed(self, wav_16k: np.ndarray):
        x = tf.convert_to_tensor(wav_16k, dtype=tf.float32)

        # Try: model(waveform)
        try:
            out = self.model(x)
        except Exception:
            # Try: model({'audio': waveform}) or {'waveform': waveform}
            tried = False
            for k in ("audio", "waveform", "inputs"):
                try:
                    out = self.model({k: x})
                    tried = True
                    break
                except Exception:
                    continue
            if not tried:
                # Try serving signature
                if hasattr(self.model, "signatures") and "default" in self.model.signatures:
                    out = self.model.signatures["default"](x)
                else:
                    raise

        # Unpack common output structures
        if isinstance(out, dict):
            for key in ("embedding", "embeddings", "outputs", "features"):
                if key in out:
                    out = out[key]
                    break

        emb = out.numpy()
        # ensure shape (T, 128) or similar
        if emb.ndim == 1:
            emb = emb[None, :]
        return emb.astype(np.float32)

    def distance(self, ref: np.ndarray, deg: np.ndarray) -> float:

        # VGGish typically expects 16 kHz mono waveform. :contentReference[oaicite:2]{index=2}
        ref16 = librosa.resample(ref, orig_sr=self.sr, target_sr=self.sr) if False else ref
        deg16 = librosa.resample(deg, orig_sr=self.sr, target_sr=self.sr) if False else deg
        # (We load at sr=self.sr in main; so these are already 16k.)

        E_ref = self._embed(ref16)
        E_deg = self._embed(deg16)

        n = min(E_ref.shape[0], E_deg.shape[0])
        if n == 0:
            return np.nan

        d = np.linalg.norm(E_ref[:n] - E_deg[:n], axis=1)  # L2 per patch
        return float(np.mean(d))


# ----------------------------
# 3) ViSQOL via CLI
# ----------------------------
def visqol_mos(
    visqol_path: str,
    ref_wav_path: str,
    deg_wav_path: str,
    mode: str = "speech",
    extra_args=None):
    """
    Calls the visqol executable and parses MOS from stdout.
    Typical usage:
      visqol --reference_file REF.wav --degraded_file DEG.wav --use_speech_mode --verbose :contentReference[oaicite:3]{index=3}
    """
    cmd = [visqol_path, "--reference_file", ref_wav_path, "--degraded_file", deg_wav_path, "--verbose"]

    if mode.lower() == "speech":
        cmd += ["--use_speech_mode"]
    elif mode.lower() == "audio":
        # audio mode is the default in some builds; keep cmd minimal
        pass
    else:
        raise ValueError("ViSQOL mode must be 'speech' or 'audio'.")

    if extra_args:
        cmd += extra_args

    p = subprocess.run(cmd, capture_output=True, text=True)
    txt = (p.stdout or "") + "\n" + (p.stderr or "")

    # Try to find a float MOS in output (varies by build).
    # Common outputs include "MOS-LQO:" or similar.
    m = re.search(r"(MOS(?:-LQO)?)[^\d]*([0-9]+\.[0-9]+)", txt, flags=re.IGNORECASE)
    if not m:
        # last resort: any float
        m2 = re.search(r"([0-9]+\.[0-9]+)", txt)
        if not m2:
            raise RuntimeError(f"Could not parse ViSQOL MOS from output:\n{txt}")
        return float(m2.group(1))
    return float(m.group(2))


# ----------------------------
# 4) RTF
# ----------------------------
def real_time_factor(process_seconds, audio_seconds):
    return float(process_seconds / audio_seconds)

def calc_pesq(sr, ref, deg, mode="wb"):
    # compute the pesq
    pesq_value = pesq(sr, ref, deg, mode)
    return pesq_value

def evaluate_dataset(ref_dir, predict_dir,sr=16000, n_fft=1024, hop=256, matlab_path=r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"):
    # Silence TF logs a bit
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    script_dir = str(os.path.join(pathlib.Path(__file__).resolve().parent, "VISQOL"))
    # matlab_path = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
    # ref_dir = pathlib.Path(r"F:\Feature\TIMIT_CAM4BWE_small\test\clean")
    # predict_dir = pathlib.Path(
    # r"F:\results\CAM4BWE\NeuronalNetworkType.mlp_x_attention_epochs_50_bs_32_lr_0.0001_nfft_1024_hop_256_declay_0.2_patience_5\predict")

    # lsd params
    lsd_sr = sr
    lsd_n_fft = n_fft
    lsd_hop = hop

    # vgg-ish params
    vgg_sr = sr
    vggish_hub = "https://tfhub.dev/google/vggish/1"
    vgg = VGGishDistance(hub_handle=vggish_hub, sr=vgg_sr)

    audio_list = os.listdir(predict_dir)
    audio_list = natsorted(audio_list)

    pesq_rm = RunningMean()
    lsd_rm = RunningMean()
    lsd_high_rm = RunningMean()
    vgg_rm = RunningMean()

    print("Compute PESQ, LSD and VGGDisatnce Score:")
    for audio in tqdm(audio_list):
        noisy_path = os.path.join(predict_dir, audio)
        clean_path = os.path.join(ref_dir, audio)

        ref, sr_c = sf.read(clean_path)
        deg, sr_n = sf.read(noisy_path)
        assert sr_c == sr
        assert sr_n == sr

        pesq_mos = calc_pesq(sr_c, ref, deg)
        lsd = LSD.lsd.log_spectral_distance(ref, deg, fs=sr, Range=(0, sr // 2))
        lsd_high = LSD.lsd.log_spectral_distance(ref, deg, fs=sr, Range=(4000, sr // 2))
        vgg_dist = vgg.distance(ref, deg)

        pesq_rm.add(pesq_mos)
        lsd_rm.add(lsd)
        lsd_high_rm.add(lsd_high)
        vgg_rm.add(vgg_dist)

    print(f"Compute Visqol Score for {pesq_rm.n} files:")
    #call visqol after the loop for performance increase otherwise matlab gets called 1000 times
    visqol_res = run_visqol_matlab_cli(
        matlab_exe=matlab_path,  # or full path to matlab.exe
        matlab_tools_dir=script_dir,
        # folder containing compute_visqol_batch.m
        base_clean_dir=ref_dir,
        nb_dir=predict_dir,
    )
    visqol_score = visqol_res[next(iter(visqol_res))]

    print("PESQ:", pesq_rm.mean)
    print("visqol_score:", visqol_score)
    print("lsd:", lsd_rm.mean)
    print("lsd_high:", lsd_high_rm.mean)
    print("vgg_dist:", vgg_rm.mean)

    return (
        pesq_rm.mean,
        visqol_score,
        lsd_rm.mean,
        lsd_high_rm.mean,
        vgg_rm.mean
    )
