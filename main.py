import numpy as np
import librosa
import soundfile as sf
from scipy.linalg import svd
import os

def run_pca_separation_from_array(x, fs, k=5, output_dir="output_files", suffix=None):
    """
    Main algorithm for PCA-based source separation.
    """
    # STAGE 1 — Signal Acquisition
    if x.ndim > 1:
        x = np.mean(x, axis=0)

    # STAGE 2 — STFT
    M, H = 1024, 256
    S = librosa.stft(x, n_fft=M, hop_length=H, window='hann')

    # STAGE 3 — Magnitude + Phase
    S_mag = np.abs(S)
    Phase = np.exp(1j * np.angle(S))

    # STAGE 4 — Centering
    mu = np.mean(S_mag, axis=1, keepdims=True)
    B = S_mag - mu

    # STAGE 5 — Truncated SVD
    U, sigma, Vt = svd(B, full_matrices=False)
    U_k = U[:, :k]
    sigma_k = sigma[:k]
    Vt_k = Vt[:k, :]
    B_voice = U_k @ np.diag(sigma_k) @ Vt_k

    # STAGE 6 — Reconstruction
    eps = 1e-10
    S_voice_mag = np.maximum(B_voice + mu, 0)
    mask = S_voice_mag / (S_mag + eps)
    mask = np.clip(mask, 0, 1)
    S_voice = S_mag * mask * Phase
    x_voice = librosa.istft(S_voice, hop_length=H, window='hann')
    min_len = min(len(x), len(x_voice))
    x_voice = x_voice[:min_len]
    x_inst = x[:min_len] - x_voice

    # STAGE 7 — Export
    if suffix is not None:
        os.makedirs(output_dir, exist_ok=True)
        sf.write(os.path.join(output_dir, f"vocal_{suffix}.wav"), x_voice, fs)
        sf.write(os.path.join(output_dir, f"inst_{suffix}.wav"), x_inst, fs)

    return x_voice, x_inst


if __name__ == "__main__":
    dataset_path = "MIR-1k(small)"
    wav_files = [f for f in os.listdir(dataset_path) if f.endswith(".wav")]
    if not wav_files:
        raise RuntimeError("No .wav files found")
    file_name = sorted(wav_files)[0]
    file_path = os.path.join(dataset_path, file_name)
    y, sr = librosa.load(file_path, sr=16000, mono=False)
    x_mixed = y[0] + y[1]
    name = file_name.replace(".wav", "")
    run_pca_separation_from_array(x_mixed, sr, k=5, output_dir="output_files", suffix=name)
    print(f"Processed: {file_name}")
