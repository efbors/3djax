import numpy as np


def fft_convolve_same(signal, filter_taps):
    """
    Convolves a batched signal with filter_taps along the last dimension using FFT.
    Returns the 'same' length output.
    """
    sig_len = signal.shape[-1]
    tap_len = filter_taps.shape[-1]

    # Linear convolution requires padding to N+M-1
    full_len = sig_len + tap_len - 1

    # JAX/XLA optimal padding: Exact next power of two
    fast_len = 2 ** np.ceil(np.log2(full_len)).astype(int)

    # Execute pure tensor FFTs along the last axis
    SIG = np.fft.fft(signal, n=fast_len, axis=-1)
    TAPS = np.fft.fft(filter_taps, n=fast_len, axis=-1)

    # Point-wise multiplication and Inverse FFT
    out = np.fft.ifft(SIG * TAPS, n=fast_len, axis=-1)

    # Trim the padding to return 'same' mode length
    start = (tap_len - 1) // 2
    end = start + sig_len

    return out[..., start:end]
