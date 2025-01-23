import dataloader_notparalell
import numpy as np
import os
import scipy.io.wavfile as wav
import torch
from models.generator import TSCNet

from models.generator import TSCNet
import torch
import os
import scipy.io.wavfile as wav
import numpy as np
from utils import power_compress, power_uncompress

def load_model(model_path):
    fft = 400
    state_dict = torch.load(model_path)
    model = TSCNet(num_channel=64, num_features=fft // 2 + 1)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def apply_ifft_and_save_wav(complex_signal, sample_rate, output_wav_file_path):
    """
    Applies an inverse Fourier transform to a complex signal and saves the result as a .wav file.

    Parameters:
    complex_signal (numpy.ndarray): The complex signal.
    sample_rate (int): The sample rate for the output .wav file.
    output_wav_file_path (str): Path to save the output .wav file.
    """
    # Apply inverse Fourier transform
    time_domain_signal = np.fft.ifft(complex_signal)

    # Convert to real part and scale to 16-bit PCM
    real_signal = np.real(time_domain_signal)
    scaled_signal = np.int16(real_signal / np.max(np.abs(real_signal)) * 32767)

    # Save as .wav file
    wav.write(output_wav_file_path, sample_rate, scaled_signal)

if __name__ == "__main__":

    test_data_path = r'F:\Feature\Edinburgh_raw_CMGAN_small'
    model_path = r'F:\results\debug\CMGAN\CMGAN_1723120648.1342444\CMGAN_epoch_0.pth'
    result_path = model_path + os.path.sep + 'pedict'

    cut_len = 32000
    batch_size = 4
    n_fft = 400
    hop = 100
    device = 'cpu'

    model = load_model(model_path)

    train_ds, test_ds = dataloader_notparalell.load_data(
        test_data_path, batch_size, 1, cut_len
    )

    for idx, batch in enumerate(test_ds):
        noisy = batch[1]
        predict = model(noisy)
        print('oki')

