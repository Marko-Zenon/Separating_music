"""
pca_separation.py — manual implementation of RPCA-based vocal separation
"""

import numpy as np

EPS = 1e-10


# 1. DFT / IDFT
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        s = 0
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            s += x[n] * np.exp(angle)
        X[k] = s

    return X


def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)

    for n in range(N):
        s = 0
        for k in range(N):
            angle = 2j * np.pi * k * n / N
            s += X[k] * np.exp(angle)
        x[n] = s / N

    return x.real


# 2. STFT / ISTFT
def stft_manual(x, n_fft=512, hop=128):
    window = np.hanning(n_fft)
    frames = []

    for i in range(0, len(x) - n_fft, hop):
        frame = x[i:i+n_fft] * window
        spectrum = dft(frame)
        frames.append(spectrum)

    return np.array(frames).T


def istft_manual(spec, n_fft=512, hop=128, length=None):
    T = spec.shape[1]
    x = np.zeros(T * hop + n_fft)
    window = np.hanning(n_fft)

    for t in range(T):
        frame = idft(spec[:, t])
        x[t*hop:t*hop+n_fft] += frame * window

    return x[:length]


# 3. Norms
def fro_norm(A):
    return np.sqrt(np.sum(A * A))


def vec_norm(v):
    return np.sqrt(np.sum(v * v))


# 4. Power Iteration
def power_iteration(A, iters=20):
    b = np.random.rand(A.shape[1])

    for _ in range(iters):
        b = A.T @ (A @ b)
        b = b / (vec_norm(b) + EPS)

    sigma = vec_norm(A @ b)
    u = (A @ b) / (sigma + EPS)

    return u, sigma, b


# 5. SVD (by deflation)
def svd_manual(A, k=20):
    m, n = A.shape
    U = []
    S = []
    V = []

    B = A.copy()

    for _ in range(k):
        u, sigma, v = power_iteration(B)

        if sigma < 1e-8:
            break

        U.append(u)
        S.append(sigma)
        V.append(v)

        # deflation (Акерта–Янга)
        B = B - sigma * np.outer(u, v)

    return np.array(U).T, np.array(S), np.array(V)


# 6. SVT (Singular Value Thresholding)
def svt_manual(X, tau, k=20):
    U, s, V = svd_manual(X, k=k)

    s_thresh = np.maximum(s - tau, 0)
    r = np.sum(s_thresh > 0)

    if r == 0:
        return np.zeros_like(X), 0

    return (U[:, :r] * s_thresh[:r]) @ V[:r, :], r


# 7. Soft threshold
def soft_threshold(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


# 8. Robust PCA (ALM)
def rpca_manual(D, lam=None, tol=1e-6, max_iter=50):
    F, T = D.shape

    if lam is None:
        lam = 1.0 / np.sqrt(max(F, T))
    norm_two = vec_norm(D.flatten())
    norm_inf = np.max(np.abs(D)) / lam

    Y = D / max(norm_two, norm_inf)

    L = np.zeros_like(D)
    S = np.zeros_like(D)

    mu = 1.25 / (norm_two + EPS)
    rho = 1.5
    d_norm = fro_norm(D)

    for _ in range(max_iter):
        L, _ = svt_manual(D - S + Y / mu, 1 / mu)
        S = soft_threshold(D - L + Y / mu, lam / mu)

        Z = D - L - S
        err = fro_norm(Z) / (d_norm + EPS)

        if err < tol:
            break

        Y = Y + mu * Z
        mu *= rho

    return L, S


# 9. Pipeline
def run_rpca_full_manual(x):
    spec = stft_manual(x)
    D = np.abs(spec)
    phase = np.exp(1j * np.angle(spec))

    L, S = rpca_manual(D)

    voice_mask = S / (S + L + EPS)

    voice_spec = D * voice_mask * phase
    inst_spec = D * (1 - voice_mask) * phase

    voice = istft_manual(voice_spec, length=len(x))
    inst = istft_manual(inst_spec, length=len(x))

    return voice, inst