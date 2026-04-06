
**Vocal Separation using Principal Component Analysis (SVD)**

This repository contains an implementation of a vocal separation algorithm based on linear algebra techniques. The project aims to isolate vocal tracks from mono mixtures by identifying dominant patterns in the frequency domain using Singular Value Decomposition (SVD).

- **Algorithm Overview**
The implementation follows the methodology detailed in the research report (Section 5: Implementation of Vocal Separation). The process consists of the following stages:

Short-Time Fourier Transform (STFT): Conversion of the time-domain signal into a magnitude spectrogram.

Magnitude Spectrogram Centering: Calculating the mean of time frames and centering the matrix to prepare for PCA.

Truncated SVD: Decomposing the centered matrix. Based on the Eckart-Young theorem, the top k singular values and their corresponding vectors capture the primary structure of the vocal signal.

Ratio Masking (Soft Masking): To minimize artifacts and preserve phase consistency, a mask is calculated based on the reconstructed vocal energy and applied to the original complex spectrogram.

Inverse STFT (ISTFT): Reconstruction of the audio signal back into the time domain.

- **Dataset: MIR-1K**
The evaluation is performed using a subset of the MIR-1K dataset (Multimedia Information Retrieval Lab).

Format: 16,000 Hz Stereo WAV files.

Channel Mapping: The left channel contains the background music (accompaniment), and the right channel contains the clean vocal (ground truth).

**Experimental Results and Discussion** 

1. Empirical Analysis of the Parameter $k$To determine the optimal number of principal components ($k$) for vocal separation, an empirical test was conducted across the dataset. The performance was measured using the Average Source-to-Distortion Ratio (SDR) in decibels (dB).The results, as visualized in the provided plot, reveal several key characteristics of the PCA-based separation:Initial Growth Phase ($k = 1$ to $10$): At low values of $k$, the SDR is significantly negative (starting at approximately $-2.0$ dB for $k=1$). This indicates that a single principal component is insufficient to reconstruct the complex spectral structure of a human voice. As $k$ increases, there is a steep improvement in quality, as more vocal nuances and frequency harmonics are captured.The "Elbow" Region ($k \approx 5$ to $10$): The curve exhibits a clear "knee" or "elbow" in this range. This confirms the hypothesis that the primary energy and structural information of the vocal signal are concentrated within the first few singular values.Saturation and Plateau ($k > 15$): Beyond $k=15$, the SDR reaches a plateau, stabilizing slightly above $0$ dB. Increasing $k$ further yields diminishing returns. This suggests that while more components are being added, they likely contain a mix of residual vocal details and instrumental "leakage," preventing further significant gains in separation purity.

2. Performance EvaluationThe maximum average SDR achieved was approximately $0.22$ dB. While this value is lower than those achieved by modern supervised deep learning models, it validates the project's objective as a successful unsupervised linear separation proof-of-concept.The results demonstrate a clear trade-off:Lower $k$: Results in a "thin" or robotic vocal sound due to the loss of spectral information.Higher $k$: Leads to better vocal reconstruction but increases the risk of "musical noise" and interference from the accompaniment.