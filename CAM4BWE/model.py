"""
@author: debes louis
"""
import tensorflow as tf
from enum import Enum
from utils import mfcc_from_waveform, mel_from_waveform
class NeuronalNetworkType(Enum):
    mlp_x_attention = 1,
    cnn_x_attention = 2,
    mlp_stft = 3,
    mlp_stft_cnn_wav_x_attetnion = 4,
    mel_mlp_x_attetnion = 5

def init_model_mlp_x_attetnion(input_1, input_2, d1, d2, d3, num_heads, embed_dim, lr):
    # Broad Oener´s network x attention
    # First modality
    x1 = tf.keras.layers.Input(shape=(input_1,), name="Feature input_1")

    x1_dense_1 = tf.keras.layers.Dense(d1, activation="relu")(x1)
    x1_dense_2 = tf.keras.layers.Dense(d2, activation="relu")(x1_dense_1)
    x1_output = tf.keras.layers.Dense(d3, activation="linear")(x1_dense_2)     # maybe relu activition
    x1_output_ext = tf.keras.layers.Reshape((d3, 1))(x1_output)

    # Second modality
    x2 = tf.keras.layers.Input(shape=(input_2,), name="Feature input_2")

    x2_dense_1 = tf.keras.layers.Dense(d1, activation="relu")(x2)
    x2_dense_2 = tf.keras.layers.Dense(d2, activation="relu")(x2_dense_1)
    x2_output = tf.keras.layers.Dense(d3, activation="linear")(x2_dense_2)     # maybe relu activition
    x2_output_ext = tf.keras.layers.Reshape((d3, 1))(x2_output)

    # X-Attention
    attn_1 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=0.2
    )

    # output of MLPs to cross attetion mechanism
    cma1 = attn_1(query=x2_output_ext, value=x1_output_ext, key=x1_output_ext)
    cma1_reduce = tf.keras.layers.Reshape((d3,))(cma1)
    # Fusion + Classification

    out_1 = tf.keras.layers.Dense(d1, activation="relu")(cma1_reduce)
    out_2 = tf.keras.layers.Dense(d2, activation="relu")(out_1)
    output = tf.keras.layers.Dense(input_2, activation="linear")(out_2)

    model = tf.keras.Model(inputs=[x1, x2], outputs=output)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])

    return model

def init_model_cnn_x_attention(waveform_length, feature_dim, lstm_units, embed_dim, num_heads, lr):
    # ============================================================================
    # Inputs
    # ============================================================================
    x1 = tf.keras.layers.Input(shape=(50, waveform_length,), name="waveform_input")
    x2 = tf.keras.layers.Input(shape=(50, feature_dim,), name="feature_input")

    # ============================================================================
    # Feature Block 1 (FB1) — waveform
    # ============================================================================
    fb1 = tf.keras.layers.Reshape((50, waveform_length, 1))(x1)         # (B, T, 1)

    fb1 = tf.keras.layers.Conv2D(64, 14, padding="same", activation="relu")(fb1)
    fb1 = tf.keras.layers.MaxPooling2D(2)(fb1)

    fb1 = tf.keras.layers.Conv2D(128, 14, padding="same", activation="relu")(fb1)
    fb1 = tf.keras.layers.MaxPooling2D(2)(fb1)

    Re_shape = tf.keras.layers.Reshape((12, 128*128))(fb1)
    fb1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True)
    )(Re_shape)                                                  # (B, seq1, 128)

    # ============================================================================
    # Feature Block 2 (FB2) — auxiliary features
    # ============================================================================
    fb2 = tf.keras.layers.Reshape((50, feature_dim, 1))(x2)              # (B, M, 1)

    fb2 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(fb2)
    fb2 = tf.keras.layers.MaxPooling2D(2)(fb2)

    fb2 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(fb2)
    fb2 = tf.keras.layers.MaxPooling2D(2)(fb2)

    Re_shape_2 = tf.keras.layers.Reshape((12, 64*128))(fb2)

    fb2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True)
    )(Re_shape_2)                                                  # (B, seq2, 128)

    # ============================================================================
    # Cross-Modal Attention (CMA)
    # ============================================================================
    attn_1 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads
    )

    attn_2 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads
    )

    # FB1 queries FB2
    cma1 = attn_1(query=fb1, value=fb2, key=fb2)            # (B, seq1, 128)
    #cma1 = tf.keras.layers.GlobalAveragePooling1D()(cma1)            # (B, 128)

    # FB2 queries FB1
    cma2 = attn_2(query=fb2, value=fb1, key=fb1)            # (B, seq2, 128)
    #cma2 = tf.keras.layers.GlobalAveragePooling1D()(cma2)            # (B, 128)

    # ============================================================================
    # Fusion + Classification
    # ============================================================================
    fused = tf.keras.layers.Concatenate()([cma1, cma2])               # (B, 256)
    x = tf.keras.layers.Reshape((12, 256, 4), name="reshape_to_2d")(fused)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    # upsample time: 12 -> 24
    x = tf.keras.layers.UpSampling2D(size=(2, 1), name="up_time_x2_1")(x)  # (None, 24, 256, C)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)

    # upsample time: 24 -> 48
    x = tf.keras.layers.UpSampling2D(size=(2, 1), name="up_time_x2_2")(x)  # (None, 48, 256, C)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    # now pad to exact target:
    # time: 48 -> 50 (pad 1 at start and 1 at end)
    # freq: 256 -> 257 (pad 1 column at the end)
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (0, 1)), name="pad_to_50_257")(x)  # (None, 50, 257, C)

    # project to 1 channel (a single magnitude value per TF bin)
    x = tf.keras.layers.Conv2D(1, 1, padding="same", name="to_single_channel")(x)  # (None, 50, 257, 1)

    # drop the channel dim -> (None, 50, 257)
    output = tf.keras.layers.Reshape((50, 257), name="output_50_257")(x)
    # ============================================================================
    # Model
    # ============================================================================
    model = tf.keras.Model(inputs=[x1, x2], outputs=output)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    return model

def mlp_stft(input_1,d1, d2, d3, lr):
    x1 = tf.keras.layers.Input(shape=(input_1,), name="Feature input_1")

    x1_dense_1 = tf.keras.layers.Dense(d1, activation="relu")(x1)
    x1_dense_2 = tf.keras.layers.Dense(d2, activation="relu")(x1_dense_1)
    x1_output = tf.keras.layers.Dense(d3, activation="linear")(x1_dense_2)  # maybe relu activition

    model = tf.keras.Model(inputs=x1, outputs=x1_output)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])

    return model


def init_model_mlp_stft_cnn_wav_x_attetnion(input_1, input_2, d1, d2, d3, c1, c2, c3, num_heads, embed_dim, lr):
    # Broad Oener´s network x attention
    # First modality
    x1 = tf.keras.layers.Input(shape=(input_1,), name="Feature input_1")
    x1_ext = tf.keras.layers.Reshape((512, 1))(x1)

    fb0 = tf.keras.layers.Conv1D(64, 15, padding="same", activation="relu")(x1_ext)
    fb0_p = tf.keras.layers.MaxPooling1D(2)(fb0)

    fb1 = tf.keras.layers.Conv1D(128, 15, padding="same", activation="relu")(fb0_p)
    fb1_p = tf.keras.layers.MaxPooling1D(2)(fb1)


    # Second modality
    x2 = tf.keras.layers.Input(shape=(input_2,), name="Feature input_2")

    x2_dense_1 = tf.keras.layers.Dense(d1, activation="relu")(x2)
    x2_dense_2 = tf.keras.layers.Dense(d2, activation="relu")(x2_dense_1)
    x2_output = tf.keras.layers.Dense(d3, activation="linear")(x2_dense_2)     # maybe relu activition
    x2_output_ext = tf.keras.layers.Reshape((d3, 1))(x2_output)

    # X-Attention
    attn_1 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=0.2
    )

    # output of MLPs to cross attetion mechanism
    cma1 = attn_1(query=x2_output_ext, value=fb1_p, key=fb1_p)
    cma1_reduce = tf.keras.layers.GlobalAveragePooling1D()(cma1)
    # Fusion + Classification

    output = tf.keras.layers.Dense(input_2, activation="linear")(cma1_reduce)

    model = tf.keras.Model(inputs=[x1, x2], outputs=output)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])

    return model

def init_model_mfccs_mlp_x_attetnion(input_1, input_2, d1, d2, d3,z_units, gru_units, num_heads, embed_dim, lr):
    # Broad Oener´s network x attention
    # First modality MFCCs

    x1 = tf.keras.layers.Input(shape=(input_1,), name="signal_wav")
    #x1_reduce = tf.keras.layers.Lambda(lambda x: x[:, :-1, :], name="drop_last_frame")(x1_mfcc)
    # InstanceNorm1d-like over time (normalize across frames for each MFCC channel, per sample)
    x1_dense_1 = tf.keras.layers.Dense(257, activation="relu")(x1)
    x1_dense_2 = tf.keras.layers.Dense(128, activation="relu")(x1_dense_1)
    x1_output = tf.keras.layers.Dense(257, activation="linear")(x1_dense_2)     # maybe relu activition
    x1_output_ext = tf.keras.layers.Reshape((d3, 1))(x1_output)

    # Second modality STFT
    x2 = tf.keras.layers.Input(shape=(input_2,), name="Feature input_2 STFT")

    x2_dense_1 = tf.keras.layers.Dense(d1, activation="relu")(x2)
    x2_dense_2 = tf.keras.layers.Dense(d2, activation="relu")(x2_dense_1)
    x2_output = tf.keras.layers.Dense(d3, activation="linear")(x2_dense_2)     # maybe relu activition
    x2_output_ext = tf.keras.layers.Reshape((d3, 1))(x2_output)

    # X-Attention
    attn_1 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=0.2
    )

    # output of MLPs to cross attetion mechanism
    cma1 = attn_1(query=x2_output_ext, value=x1_output_ext, key=x1_output_ext)
    cma1_reduce = tf.keras.layers.Reshape((d3,))(cma1)
    # Fusion + Classification

    out_1 = tf.keras.layers.Dense(d1, activation="relu")(cma1_reduce)
    out_2 = tf.keras.layers.Dense(d2, activation="relu")(out_1)
    output = tf.keras.layers.Dense(input_2, activation="linear")(out_2)

    model = tf.keras.Model(inputs=[x1, x2], outputs=output)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])

    return model