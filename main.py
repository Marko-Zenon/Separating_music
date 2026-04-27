import numpy as np
import librosa
import soundfile as sf
from scipy.linalg import svd
import os


def run_pca_separation_from_array(x, fs, k_vocal=100, k_skip=0,
                                   output_dir="output_files", suffix=None):
    if x.ndim > 1:
        x = np.mean(x, axis=0)          # x ∈ ℝᴺ

    M, H = 1024, 256
    S = librosa.stft(x, n_fft=M, hop_length=H, window='hann')  # ℂ^{513×T}

    eps = 1e-10
    S_mag = np.abs(S)                                           # ℝ^{513×T}
    Phase = np.exp(1j * np.angle(S))                           # ℂ^{513×T}

    mu = np.mean(S_mag, axis=1, keepdims=True)                 # ℝ^{513×1}
    B  = S_mag - mu                                            # ℝ^{513×T}

    U, sigma, Vt = svd(B, full_matrices=False)                 # economy SVD

    # Vocal subspace: components 0 … k_vocal-1
    B_voice = (U[:, :k_vocal]
               @ np.diag(sigma[:k_vocal])
               @ Vt[:k_vocal, :])                              # ℝ^{513×T}

    S_voice_approx = np.maximum(B_voice + mu, 0)              # ℝ^{513×T}≥0

    mask_voice = np.clip(S_voice_approx / (S_mag + eps), 0.0, 1.0)

    mask_inst = 1.0 - mask_voice

    S_voice_out = S_mag * mask_voice * Phase                   # ℂ^{513×T}
    S_inst_out  = S_mag * mask_inst  * Phase                   # ℂ^{513×T}

    x_voice = librosa.istft(S_voice_out, hop_length=H, window='hann')
    x_inst  = librosa.istft(S_inst_out,  hop_length=H, window='hann')

    min_len = min(len(x), len(x_voice), len(x_inst))
    x_voice = x_voice[:min_len]
    x_inst  = x_inst[:min_len]

    if suffix is not None:
        os.makedirs(output_dir, exist_ok=True)
        sf.write(os.path.join(output_dir, f"vocal_{suffix}.wav"), x_voice, fs)
        sf.write(os.path.join(output_dir, f"inst_{suffix}.wav"),  x_inst,  fs)

    return x_voice, x_inst

if __name__ == "__main__":
    dataset_path = "papka"
    wav_files = [f for f in os.listdir(dataset_path) if f.endswith(".wav")]
    if not wav_files:
        raise RuntimeError("No .wav files found in " + dataset_path)

    file_name = sorted(wav_files)[0]
    file_path = os.path.join(dataset_path, file_name)

    y, sr = librosa.load(file_path, sr=16000, mono=False)
    x_mixed = y[0] + y[1]                    # sum both stereo channels

    name = file_name.replace(".wav", "")
    x_v, x_i = run_pca_separation_from_array(
        x_mixed, sr,
        k_vocal=50, k_skip=100,
        output_dir="output_files",
        suffix=name
    )
    print(f"Processed: {file_name}  |  vocal len={len(x_v)}  inst len={len(x_i)}")
    print("fdjshgfsdkj")