import numpy as np
from scipy.signal import stft

def freq2bin(freq_range_hz, fmax_hz, n_bins):
    """
    MATLAB-like freq2bin: map [f_low, f_high] in Hz to bin indices.
    Bins are assumed to be linearly spaced from 0..fmax with n_bins points.
    Returns (i0, i1) inclusive indices.
    """
    f_low, f_high = freq_range_hz
    freqs = np.linspace(0.0, fmax_hz, n_bins)

    i0 = np.searchsorted(freqs, f_low, side="left")
    i1 = np.searchsorted(freqs, f_high, side="right") - 1

    i0 = int(np.clip(i0, 0, n_bins - 1))
    i1 = int(np.clip(i1, 0, n_bins - 1))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1


def my_specgram(x, fs, W, SP, nfft=None):
    """
    Replacement for mySpecgram(x, W, SP).
    Assumption: SP is the *overlap fraction* (e.g., 0.4 -> 40% overlap).
    """
    x = np.asarray(x).reshape(-1)

    nperseg = int(W)
    noverlap = int(round(SP * nperseg))
    if nfft is None:
        nfft = nperseg

    f, t, Z = stft(
        x,
        fs=fs,
        window="hamm",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )
    return Z  # shape: (freq_bins, frames)


def log_spectral_distance(Clean, Noisy, fs, Range):
    # Len=min(length(Clean),length(Noisy));
    Clean = np.asarray(Clean).reshape(-1)
    Noisy = np.asarray(Noisy).reshape(-1)

    Len = min(Clean.size, Noisy.size)
    Clean = Clean[:Len]
    Noisy = Noisy[:Len]

    # Clean=Clean./sqrt(sum(Clean.^2)); etc.
    eps = 1e-12
    clean_norm = np.sqrt(np.sum(Clean**2))
    noisy_norm = np.sqrt(np.sum(Noisy**2))
    Clean = Clean / max(clean_norm, eps)
    Noisy = Noisy / max(noisy_norm, eps)

    # W=round(.025*fs); SP=.4;
    W = int(round(0.025 * fs))
    SP = 0.4

    # CL=abs(mySpecgram(...)); NO=abs(mySpecgram(...));
    CL = np.abs(my_specgram(Clean, fs, W, SP))
    NO = np.abs(my_specgram(Noisy, fs, W, SP))

    # nfft2=size(CL,1);
    nfft2 = CL.shape[0]

    # N=min(size(CL,2),size(NO,2));
    N = min(CL.shape[1], NO.shape[1])

    # RangeBin=freq2bin(Range,fs/2,nfft2); RangeBin=RangeBin(1):RangeBin(2);
    i0, i1 = freq2bin(Range, fs / 2.0, nfft2)
    bins = slice(i0, i1 + 1)

    # LSD=mean(sqrt(mean((log(CL(...))-log(NO(...))).^2)));
    # MATLAB: inner mean over freq bins (rows) -> per-frame, then mean over frames
    CL_sel = np.maximum(CL[bins, :N], eps)
    NO_sel = np.maximum(NO[bins, :N], eps)

    per_frame = np.sqrt(np.mean((np.log(CL_sel) - np.log(NO_sel)) ** 2, axis=0))
    LSD = float(np.mean(per_frame))
    return LSD
