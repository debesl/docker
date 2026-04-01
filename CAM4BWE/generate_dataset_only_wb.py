import os
import pathlib
from tqdm import tqdm
import torch
import soundfile as sf

def path_maker(reference_path,destination_path):
    path = []
    train_path = reference_path.joinpath('train')
    valid_path = reference_path.joinpath('valid')
    test_path = reference_path.joinpath('test')

    path.append(train_path)
    path.append(valid_path)
    path.append(test_path)
    destination = [destination_path.joinpath(path[0].name), destination_path.joinpath(path[0].name), destination_path.joinpath(path[2].name)]

    return path, destination

def get_all_audio_files(path, number_of_files=None):
    wav_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))

                # stop as soon as we reach the limit
                if number_of_files is not None and len(wav_files) >= number_of_files:
                    return wav_files
    return wav_files

def gen_data(clean_files, output_directory, target_sr, cutoff_frequency):
    if not os.path.exists(os.path.join(output_directory, "clean")):
        os.makedirs(os.path.join(output_directory, "clean"))

    for clean_file in tqdm(clean_files):
        filename = os.path.splitext(os.path.basename(clean_file))[0] + ".wav"
        clean_audio, sr_c = sf.read(clean_file)

        clean_output_path = os.path.join(output_directory, "clean", filename)
        save_wav_soundfile(clean_output_path, clean_audio, target_sr)

def save_wav_soundfile(path, audio_tensor, sample_rate):
    # torchaudio tensors are usually [channels, time]
    if isinstance(audio_tensor, torch.Tensor):
        audio = audio_tensor.detach().cpu()
        if audio.dim() == 2:
            audio = audio.transpose(0, 1)  # -> [time, channels]
        audio = audio.numpy()
    else:
        audio = audio_tensor

    sf.write(path, audio, sample_rate, subtype="PCM_16")

if __name__ == "__main__":

    data_path = pathlib.Path("F:\Feature\TIMIT")
    destination_path = pathlib.Path("F:\Feature\TIMIT_CAM4BWE")
    number_of_files = 100000                                #just for debugging, set None to process all
    target_sr = 16000
    cutoff_frequency = 4000

    reference, destination = path_maker(reference_path=data_path, destination_path=destination_path)

    for idx, elem in enumerate(reference):
        if 'test' in str(elem):
            print("Generating Test set:")
            audio_files = get_all_audio_files(elem, number_of_files=number_of_files)

            gen_data(clean_files=audio_files, output_directory=destination[2], target_sr=target_sr, cutoff_frequency=cutoff_frequency)
        elif 'train' in str(elem):
            print("Generating train set:")
            audio_files = get_all_audio_files(elem, number_of_files=number_of_files)

            gen_data(clean_files=audio_files, output_directory=destination[0], target_sr=target_sr, cutoff_frequency=cutoff_frequency)


        if "valid" in str(elem):
            print("Generating valid set:")
            audio_files = get_all_audio_files(elem, number_of_files=number_of_files)

            gen_data(clean_files=audio_files, output_directory=destination[1], target_sr=target_sr, cutoff_frequency=cutoff_frequency)
