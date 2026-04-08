import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from mir_eval.separation import bss_eval_sources
from main import run_pca_separation_from_array 

def empirical_k_analysis(dataset_path, k_range):
    """
    Emperical analysis to determine optimal k for PCA-based source separation on the MIR-1K dataset.
    """
    wav_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
    total_files = len(wav_files)
    results = {k: [] for k in k_range}
    print(f"Amount of files to analyze: {total_files}")

    for i, file_name in enumerate(wav_files):
        file_path = os.path.join(dataset_path, file_name)
        try:
            y, sr = librosa.load(file_path, sr=16000, mono=False)
            x_mixed = np.mean(y, axis=0) 
            vocal_ref = y[1, :] 
            for k in k_range:
                x_voice, _ = run_pca_separation_from_array(x_mixed, sr, k=k, suffix=None)
                min_len = min(len(vocal_ref), len(x_voice))
                ref = vocal_ref[:min_len].reshape(1, -1)
                est = x_voice[:min_len].reshape(1, -1)
                sdr, _, _, _ = bss_eval_sources(ref, est, compute_permutation=False)
                results[k].append(sdr[0])
            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                print(f"Progress: {i + 1}/{total_files} files processed...")
        except Exception as e:
            print(f"Error in file {file_name}: {e}")
    avg_sdr = [np.mean(results[k]) for k in k_range]

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, avg_sdr, 'bo-', linewidth=2, markersize=8)
    plt.xticks(list(k_range))
    plt.title("Analysis complete for k values")
    plt.xlabel("k (number of principal components)")
    plt.ylabel("Average SDR (dB)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('final_empirical_results.png')
    plt.show()

    best_k = k_range[np.argmax(avg_sdr)]
    print(f"\nAnalysis complete")
    print(f"Optimal k: {best_k}")
    print(f"Maximum average SDR: {max(avg_sdr):.2f} dB")

if __name__ == "__main__":
    path_to_wavs = "MIR-1k(small)"
    empirical_k_analysis(path_to_wavs, k_range=range(1, 30))

# Optimal k: 20
# Maximum average SDR: 0.23 dB
