"""
@author: debes louis
"""
import tensorflow as tf
import os
from model import NeuronalNetworkType, init_model_mlp_x_attetnion, init_model_cnn_x_attention, mlp_stft, \
    init_model_mlp_stft_cnn_wav_x_attetnion, init_model_mfccs_mlp_x_attetnion
from utils import callback_functions
from DataGenerator import generate_dataset_training, build_frames

# ============================================================================
# GPU / CUDA Fuck
# ============================================================================
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" #enable if memory overflow druing training / validation

cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"], cross_device_ops=cross_device_ops) # set stragy to init model distributed on both gpus
#strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0"], cross_device_ops=cross_device_ops) # set stragy to init model distributed on both gpus

# tf.debugging.set_log_device_placement(True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_physical_devices('GPU')
        print(len(gpus), 'Physical GPUs', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

def main():

    # ============================================================================
    # Global Parameters
    # ============================================================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "clean")
    # currrently only Timit with Train/noisy clean gets supported
    #model_type = NeuronalNetworkType.cnn_x_attention  # choose model you would like to train
    model_type = NeuronalNetworkType.mlp_stft
    # ============================================================================
    #  Data preProcessing
    # ============================================================================
    # Signal / STFT parameters
    # ========================
    sr = 16000  # target sampling rate
    n_fft = 512
    hop_length = 256  # window hop for fft computation
    waveform_len = 512  # temporal context (NOT equal to n_fft) input shape 1 of modality time
    feature_dim = (n_fft // 2) + 1  # input shape 2 of modality TF-domain (STFT-Spectrum)
    #num_frames = 50
    num_frames = None
    # ============================================================================
    #  Trianing Prameters
    # ============================================================================
    lr = 0.001 # learning rate for ADAM optimizer
    lr_decay_factor = 2e-5  # factor how the learing rate gets reduced
    patience = 5  # number of epochs, the learning rate is reduced.
    cut_off = 4000
    number_of_files = 10
    epochs = 50
    train_batch_size = 64       # bs for each gpu
    #result_foder = fr"F:\results\CAM4BWE\{str(model_type)[20:]}_cnn2d_decoder_epochs_{epochs}_bs_{train_batch_size}_lr_{lr}_nfft_{n_fft}_hop_{hop_length}_decay_{lr_decay_factor}_patience_{patience}"
    result_foder = "results"
    if not os.path.exists(result_foder):
        os.makedirs(result_foder, exist_ok=True)

    # ============================================================================
    # Bild Dataset
    # ============================================================================
    print("Building Dataset.... This takes a while, get a coffee or two:")

    X_wave, X_low_stft, Y_high_stft = generate_dataset_training(data_root,
                                                                hop_length,
                                                                n_fft,
                                                                waveform_len,
                                                                sr,
                                                                cut_off=cut_off,
                                                                number_of_files=number_of_files)
    #if num_frames:
    #    X_wave = build_frames(X_wave, num_frames)
    #    X_low_stft = build_frames(X_low_stft, num_frames)
     #   Y_high_stft = build_frames(Y_high_stft, num_frames)

    # ============================================================================
    # Build Model
    # ============================================================================
    with strategy.scope():
        if model_type == NeuronalNetworkType.mlp_stft_cnn_wav_x_attetnion:
            embed_dim = 128  # 2 * lstm_units
            num_heads = 8  # cross attention heads
            model = init_model_mlp_stft_cnn_wav_x_attetnion(input_1=waveform_len,
                                                            input_2=feature_dim,
                                                            d1=feature_dim, d2=feature_dim // 2, d3=feature_dim,
                                                            c1=32, c2=64, c3=257,
                                                            embed_dim=embed_dim,
                                                            num_heads=num_heads,
                                                            lr=lr)
            train_ds = [X_wave, X_low_stft]

        if model_type == NeuronalNetworkType.cnn_x_attention:
            lstm_units = 64
            embed_dim = 128  # 2 * lstm_units
            num_heads = 8  # cross attention heads
            model = init_model_cnn_x_attention(waveform_length=waveform_len,
                                               feature_dim=feature_dim,
                                               lstm_units=lstm_units,
                                               embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               lr=lr)
            train_ds = [X_wave, X_low_stft]

        if model_type == NeuronalNetworkType.mlp_x_attention:
            embed_dim = 128  # attetion neurons
            num_heads = 8  # cross attention heads
            model = init_model_mlp_x_attetnion(input_1=waveform_len,
                                               input_2=feature_dim,
                                               d1=feature_dim, d2=feature_dim // 2, d3=feature_dim,
                                               embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               lr=lr)
            train_ds = [X_wave, X_low_stft]

        if model_type == NeuronalNetworkType.mlp_stft:

            model = mlp_stft(input_1=feature_dim,
                             d1=feature_dim,
                             d2=feature_dim // 2,
                             d3=feature_dim,
                             lr=lr)

            train_ds = X_low_stft

    # ============================================================================
    # Callback Functions
    # ============================================================================
    callbacks = callback_functions(result_foder, lr_decay_factor, patience)

    # ============================================================================
    # Training
    # ============================================================================

    history = model.fit(train_ds,
                        y=Y_high_stft,
                        validation_split=0.2,
                        batch_size=train_batch_size,
                        epochs=epochs,
                        callbacks=[callbacks])
    return history

if __name__ == "__main__":

    main()




