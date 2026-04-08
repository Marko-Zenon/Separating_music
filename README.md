# Vocal Separation using PCA-Based Matrix Decomposition

This project implements monaural source separation to isolate lead vocals from instrumental accompaniment in music recordings. The approach utilizes Principal Component Analysis (PCA) via Singular Value Decomposition (SVD) to exploit the structured, low-rank nature of accompaniment signals, while vocals are modeled as higher-rank residual components.

## Algorithm Methodology

The system processes audio through a seven-stage pipeline grounded in linear algebra:

1.  **Mono Conversion:** Stereo recordings are downmixed to a single channel by taking the row-wise mean of the signal matrix.
2.  **Short-Time Fourier Transform (STFT):** The 1D signal is transformed into a 2D time-frequency spectrogram using a window size ($M$) of 1024 and a hop size ($H$) of 256 samples.
3.  **Magnitude and Phase Decomposition:** The complex spectrogram $S$ is split into its magnitude ($S_{mag}$) for processing and its phase for later reconstruction.
4.  **Centering:** To ensure PCA captures variance rather than the average spectrum, the per-frequency mean is subtracted from each row of the magnitude spectrogram.
5.  **Truncated SVD:** An economy SVD is performed on the centered matrix. Empirical analysis suggests that performance saturates around k ≈ 15–20.
6.  **Soft Mask Reconstruction:** To reduce audible artifacts, a Wiener-like soft ratio mask is calculated and applied to the original spectrogram before performing the Inverse STFT (ISTFT).
7.  **Export:** The separated vocal and instrumental (residual) tracks are written to the disk as WAV files.



## Dataset and Evaluation

The algorithm is benchmarked using the **MIR-1K** dataset(but in repository we have a smaller version of it, because of the size), which contains 1,000 song clips of Mandarin pop music.

* **Ground Truth:** Recordings provide the vocal and instrumental tracks on separate channels, allowing for precise quantitative evaluation.
* **Metric:** Performance is measured using the **Source-to-Distortion Ratio (SDR)** in decibels (dB).
* **Hyperparameter Selection:** Empirical analysis across the full dataset determined that **$k=20$** is the optimal number of principal components.
* **Results:** The system achieved an average SDR of approximately **0.23 dB**, which is consistent for an unsupervised, assumption-light method.

## Structure of project

main.py - a file with an algorithm that performs the main task
analysis.py - a file with testing different value for k


### Prerequisites
Install the required libraries:
```bash
pip install numpy librosa scipy soundfile mir_eval matplotlib
