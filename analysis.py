"""
analysis.py — minimal RPCA evaluation on MIR-1k(small)
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from pca_separation import run_rpca_full_manual

import mir_eval

DATASET_DIR = "MIR-1k(small)"
OUTPUT_DIR = "analysis_out"

LAMBDA_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5]
SR = 16000


def load_file(path):
    audio, sr = sf.read(path, always_2d=True)
    inst = audio[:, 0]
    voice = audio[:, 1]
    mix = inst + voice
    if sr != SR:
        mix = librosa.resample(mix, sr, SR)
        voice = librosa.resample(voice, sr, SR)
        inst = librosa.resample(inst, sr, SR)
    return mix, voice, inst


def compute_sdr(v_ref, i_ref, v_est, i_est):
    L = min(len(v_ref), len(v_est), len(i_ref), len(i_est))
    refs = np.stack([v_ref[:L], i_ref[:L]])
    ests = np.stack([v_est[:L], i_est[:L]])
    sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
        refs, ests, compute_permutation=False
    )
    return sdr[0], sdr[1]


def estimate_lambda(length):
    return 1 / np.sqrt(length)


def main():
    base = Path(__file__).parent
    data_path = base / DATASET_DIR
    out_path = base / OUTPUT_DIR
    out_path.mkdir(exist_ok=True)
    files = sorted(data_path.glob("*.wav"))
    print(f"Files: {len(files)}\n")
    results = []
    for mult in LAMBDA_MULTS:
        print(f"λ multiplier = {mult}")
        sdr_voice = []
        sdr_inst = []

        for f in tqdm(files):
            try:
                mix, v_ref, i_ref = load_file(f)
                lam = mult * estimate_lambda(len(mix))
                v_est, i_est = run_rpca_full_manual(
                    mix, sr=SR, lam=lam
                )
                v_sdr, i_sdr = compute_sdr(v_ref, i_ref, v_est, i_est)
                sdr_voice.append(v_sdr)
                sdr_inst.append(i_sdr)
            except Exception:
                continue

        mean_v = np.mean(sdr_voice)
        mean_i = np.mean(sdr_inst)
        print(f"  vocal SDR: {mean_v:.3f}")
        print(f"  inst  SDR: {mean_i:.3f}\n")
        results.append((mult, mean_v, mean_i))
    best = max(results, key=lambda x: x[1])
    print("\nBest result:")
    print(f"λ multiplier = {best[0]}")
    print(f"Vocal SDR = {best[1]:.3f}")
    print(f"Inst  SDR = {best[2]:.3f}")


if __name__ == "__main__":
    main()