1. Research Area and Motivation
The automatic separation of audio sources from a single mixed recording — the cocktail-party problem — is a long-standing challenge in signal processing. In practical settings such as music production, speech enhancement, and telecommunication, a monaural (single-channel) recording captures multiple simultaneous sources: vocals, instruments, and ambient noise. This project addresses that problem directly through linear algebra, making it a natural application of eigendecomposition, spectral analysis, and matrix factorisation.
2. Aim and Objectives
The primary aim is to develop a working PCA-based vocal–instrumental separator that takes a monaural audio file as input and produces two output tracks: a vocal-dominant signal and an instrumental-dominant signal. Specific objectives are:
Digitise and frame the input audio signal using standard sampling techniques.
Convert the time-domain signal into a two-dimensional time-frequency representation via the Short-Time Fourier Transform (STFT).
Apply PCA — implemented through Truncated SVD — to the resulting spectrogram matrix to isolate dominant frequency patterns corresponding to vocal components.
Reconstruct the separated audio channels from the filtered spectral representation using the Inverse STFT (ISTFT).
Evaluate separation quality using standard quantitative metrics (BSS Eval: SDR, SIR, SAR).
3. Literature Review and Chosen Approach
Source separation has been approached through Independent Component Analysis (ICA), which requires as many recordings as sources and is inapplicable in the monaural case. Non-negative Matrix Factorisation (NMF) decomposes a spectrogram into spectral bases and temporal activations and has shown strong results in music separation. Deep learning models (U-Net, recurrent networks) achieve state-of-the-art performance but require large labelled datasets and significant computational resources. PCA-based approaches are unsupervised, computationally tractable, and grounded directly in the linear algebra methods studied in this course, making them the natural choice for this project.
4. PCA and SVD: Relationship and Joint Use
This section directly addresses the core mathematical question of the project: what is the relationship between PCA and SVD, why are both relevant, and how do they work together in our pipeline?
4.1 What PCA Achieves
Principal Component Analysis finds a new orthonormal basis for the data such that the first basis vector (principal component) points in the direction of maximum variance, the second in the direction of maximum remaining variance orthogonal to the first, and so on. Applied to our centred spectrogram matrix B (frequencies x time frames), PCA identifies the spectral patterns that account for the most energy in the signal. The lead vocal, being the most structured and energetically dominant source in a typical song recording, is expected to appear in the first few principal components. The diffuse instrumental background, being less structured, is spread across many lower-variance components.
Formally, the principal components are the eigenvectors of the covariance matrix C = (1/N) * B * B^T. The projection of B onto the top-k eigenvectors gives the best rank-k approximation of the data in the least-squares sense (Eckart–Young theorem), which is exactly the filtered vocal spectrogram we seek.
4.2 Why Classical PCA Fails Here
The covariance matrix C = B * B^T has dimensions (F x F), where F is the number of frequency bins. For a 5-second clip at 16,000 Hz with a 512-point FFT, F = 257, which is manageable — but the operation of squaring the matrix B squares its condition number: if B has condition number kappa, then B * B^T has condition number kappa^2. This dramatically amplifies rounding errors accumulated during floating-point arithmetic. For longer recordings or higher frequency resolution, the matrix becomes very large and the eigendecomposition of C is both slow and numerically unstable. Classical PCA therefore becomes impractical as a direct computational method.
4.3 SVD as the Numerically Stable Implementation of PCA
Singular Value Decomposition decomposes the centred data matrix B directly, without forming C:
B = U * Sigma * V^T
where U (F x F) is an orthogonal matrix whose columns are the left singular vectors, Sigma (F x N) is a diagonal matrix of non-negative singular values sigma_1 >= sigma_2 >= ... >= 0 sorted in descending order, and V^T (N x N) is an orthogonal matrix whose rows are the right singular vectors.
The mathematical equivalence with PCA follows directly from the relationship between the SVD of B and the eigendecomposition of C:
C = B * B^T = (U * Sigma * V^T) * (V * Sigma^T * U^T) = U * (Sigma * Sigma^T) * U^T
This shows that the columns of U are precisely the eigenvectors of the covariance matrix C, and the eigenvalues of C equal the squared singular values: lambda_i = sigma_i^2. SVD therefore computes the same principal components as classical PCA, but operates directly on B rather than on the ill-conditioned product B * B^T. This is not a workaround or an approximation — it is mathematically identical, and numerically far superior.
4.4 Truncated SVD: Why It Suffices and What It Gains
Full SVD retains all singular values and is equivalent to lossless PCA. For our application, we only need the top-k components. Truncated SVD computes only the k largest singular values and their corresponding singular vectors:
B_voice = U_k * Sigma_k * V_k^T
where U_k is the (F x k) submatrix of U, Sigma_k is the (k x k) top-left submatrix of Sigma, and V_k^T is the (k x N) submatrix of V^T. By the Eckart–Young theorem, this is the optimal rank-k approximation of B in both the Frobenius norm and the spectral norm — meaning no other rank-k matrix is closer to B. Truncated SVD is also far more efficient to compute than full SVD; using randomised algorithms (e.g., the randomised SVD of Halko et al., 2011), it runs in O(F * N * k) time rather than O(F * N * min(F, N)).
In summary, PCA defines the goal (find the k most energetic spectral directions), SVD provides the stable and efficient computational method to achieve it, and Truncated SVD restricts computation to only the components we actually need. The three are not competing alternatives — they form a coherent pipeline.
5. Pseudocode of the Implementation
The following pseudocode specifies each algorithmic step with explicit reference to the underlying linear algebra operations. For each stage we also explain why the step is necessary and what mathematical property it exploits.
INPUT:  audio file path, k (number of components to retain)
OUTPUT: x_voice(t), x_instrumental(t)

STAGE 1 — Signal Acquisition
x = load_audio(path)           // read waveform as 1-D real vector
x = resample(x, target=16000)  // standardise sampling rate f_s
x = to_mono(x)                 // average channels if stereo
Why: Establishes a uniform discrete representation. The signal is now a column vector x in R^N where N = duration * f_s.
STAGE 2 — Short-Time Fourier Transform (STFT)
Choose frame length M, hop size H (e.g. M=512, H=128)
For each time frame t = 0, 1, ..., T-1:
    x_t = x[t*H : t*H + M] * window   // extract and apply Hann window
    w_t = F * x_t                      // DFT: multiply by Fourier matrix F
    F_{n,k} = exp(-j * 2*pi * k * n / M)   // entries of F
S = [w_0 | w_1 | ... | w_{T-1}]       // assemble spectrogram (F x T matrix)
Why: The STFT converts the 1-D time signal into a 2-D matrix S in C^(F x T) where each column is a frequency-domain snapshot. This is essential because PCA requires a matrix, not a vector, and because vocal and instrumental components are more separable in the frequency domain than in the time domain.
STAGE 3 — Work on Magnitude Spectrogram
S_mag = |S|                            // element-wise absolute value
Phase = S / S_mag                      // store original phase for reconstruction
Why: SVD operates on real matrices. The magnitude captures energy distribution (which PCA exploits); the phase is preserved separately and reapplied at reconstruction to avoid artefacts.
STAGE 4 — Centring (Required for Correct PCA)
mu = mean(S_mag, axis=1)               // per-frequency mean: vector in R^F
B = S_mag - mu[:, newaxis]             // broadcast subtraction: B in R^(F x T)
Why: PCA finds directions of maximum variance around the mean. Without centring, the first principal component would point towards the data mean rather than the direction of greatest variance, corrupting the decomposition.
STAGE 5 — Truncated SVD (PCA via SVD)
U_k, sigma_k, Vt_k = truncated_svd(B, k)
// U_k in R^(F x k): top-k left singular vectors (= principal directions)
// sigma_k in R^k:   top-k singular values, sigma_1 >= sigma_2 >= ... >= sigma_k
// Vt_k in R^(k x T): top-k right singular vectors (temporal activations)
B_voice = U_k * diag(sigma_k) * Vt_k  // rank-k approximation of B
Why: By the Eckart-Young theorem, B_voice is the closest rank-k matrix to B in the Frobenius norm. The largest singular values capture the most energetic and structured patterns — expected to correspond to the vocal. SVD computes this without forming the unstable covariance matrix B*B^T.
STAGE 6 — Reconstruction
S_voice_mag = B_voice + mu[:, newaxis]  // restore mean
S_voice = S_voice_mag * Phase           // reapply original phase
x_voice = ISTFT(S_voice)               // inverse STFT -> time domain
x_instrumental = x - x_voice           // residual = background
Why: Adding back the mean reverses the centring from Stage 4, restoring absolute energy levels. Multiplying by the original phase allows ISTFT to produce a valid real audio signal. The instrumental track is obtained by simple subtraction, which is valid because the original signal is the sum of all sources.
STAGE 7 — Export
write_wav('vocal.wav', x_voice, f_s)
write_wav('instrumental.wav', x_instrumental, f_s)

6. Testing Strategy and Data
We will evaluate separation quality using BSS Eval metrics: Source-to-Distortion Ratio (SDR), Source-to-Interference Ratio (SIR), and Source-to-Artefact Ratio (SAR). These require reference clean-vocal tracks for comparison. Two data sources will be used:
MUSDB18 dataset — professionally recorded multi-track songs with isolated vocal stems, used as ground truth for quantitative evaluation.
Self-recorded clips — short vocal-over-instrumental samples for rapid, reproducible testing during development.
All audio is processed at 16,000 Hz. The implementation is written in Python using NumPy (matrix operations), SciPy (STFT/ISTFT), and librosa (audio I/O).
7. Plan of Future Research
Weeks 1–2 (current): Literature review, project setup, STFT pipeline implementation.
Weeks 3–4: Truncated SVD separation; qualitative tests; parameter tuning (M, H, k).
Weeks 5–6: Quantitative evaluation using BSS Eval metrics; comparison with a baseline.
Week 7: Final analysis, report writing, and video presentation preparation.
8. Potential Challenges
Choice of k: Too few components discard vocal content; too many retain background noise. We will select k empirically by inspecting the singular value decay curve (the point of the "elbow" in the spectrum).
Vocal dominance assumption: PCA captures statistically dominant patterns. In heavily orchestrated recordings, instruments may dominate the top singular values, causing misidentification. This is a fundamental limitation of unsupervised separation.
Phase reconstruction: Operating on the magnitude spectrogram requires reusing the original phase during ISTFT, which may introduce audible artefacts. We will investigate whether processing the full complex spectrogram mitigates this.
Computational cost: Large spectrograms demand efficient memory management. Randomised Truncated SVD will be used to keep runtime tractable for longer recordings.
