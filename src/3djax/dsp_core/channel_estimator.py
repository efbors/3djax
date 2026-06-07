import numpy as np
from scipy.linalg import convolution_matrix


class ChannelEstimator:
    def __init__(self):
        pass

    def estimate_channel_fd_zf(self, rx_am, tx_ref, tap_count=100):
        """
        Frequency Domain Zero-Forcing Channel Estimation.
        rx_am: Extracted Rx ZC sequence [batch_size, am_len]
        tx_ref: Ideal Tx ZC sequence [am_len]
        """
        tx_ref = np.squeeze(tx_ref)

        # FFT of received and reference
        Y = np.fft.fft(rx_am, axis=-1)
        X = np.fft.fft(tx_ref, axis=-1)

        # Simple division (safe because X has no nulls for ZC sequences)
        H_est = Y / X

        # IFFT to get time-domain impulse response
        h_full = np.fft.ifft(H_est, axis=1)

        # Slice only the valid causal taps, completely discarding the wrap-around boundary errors
        h_est = h_full[:, :tap_count]

        h_roll = self.extract_centered_taps(h_est, pre_taps=10, post_taps=90)

        return h_roll

    def estimate_channel_td_corr(self, rx_am, tx_ref):
        """
        Time Domain Cross-Correlation Channel Estimation.
        tx_ref at Fdac;
        """
        # Matched filter is the time-reversed complex conjugate
        tx_ref = np.squeeze(tx_ref)
        matched_filter = np.conj(tx_ref[::-1])

        # Perform cross-correlation (using the existing fft_convolve_same)
        # Note: Depending on padding, you may use standard np.convolve to see the full tail
        h_est = np.zeros_like(rx_am)
        for i in range(rx_am.shape[0]):
            h_est[i] = np.convolve(rx_am[i], matched_filter, mode='same')

        # Normalize by ZC energy
        h_est /= np.sum(np.abs(tx_ref) ** 2)

        h_roll = self.extract_centered_taps(h_est, pre_taps=10, post_taps=90)

        return h_roll

    def estimate_channel_td_ls(self, rx_am, tx_ref, tap_count=64):
        """
        Time Domain Least Squares (Optimal MMSE equivalent).
        tap_count: The number of taps you want in your estimated impulse response.

        An MMSE FFE finds taps by solving $R_{xx} w = R_{xy}$. For channel estimation,
        flip the perspective:    Build a Toeplitz convolution matrix $\mathbf{X}$ from the
        reference signal and mathematically solve for the optimal
        vector $\mathbf{h}$ that minimizes the squared error

         $||\mathbf{X}\mathbf{h} - \mathbf{y}||^2$.
         This eliminates boundary errors caused by linear convolution
         and yields the mathematically optimal linear channel estimate.
        """
        batch_size = rx_am.shape[0]
        tx_ref = np.squeeze(tx_ref)

        # 1. Build the convolution matrix X
        # X row 0 corresponds to time index (tap_count - 1) of the tx_ref
        X = convolution_matrix(tx_ref, tap_count, mode='valid')
        valid_len = X.shape[0]

        h_est_batch = np.zeros((batch_size, tap_count), dtype=np.complex128)

        # 2. Slice y to correctly align with X
        # We offset by center_delay so the estimator captures precursor taps.
        center_delay = 32
        start_idx = (tap_count - 1) - center_delay

        for i in range(1):
            # Slice the received signal to perfectly match the Toeplitz rows
            y_valid = rx_am[i, start_idx: start_idx + valid_len]

            # Solve X * h = y
            h_est, residuals, rank, s = np.linalg.lstsq(X, y_valid, rcond=None)
            h_est_batch[i] = h_est

        return h_est_batch

    def extract_centered_taps(self, h_full, pre_taps=16, post_taps=48):
        """
        Extracts the linear impulse response from the circular ZF IFFT array.
        """
        # 1. Grab the precursor from the very end of the array
        precursor = h_full[:, -pre_taps:]

        # 2. Grab the main peak (index 0) and the postcursor
        postcursor = h_full[:, :post_taps]

        # 3. Concatenate them into a single linear filter
        # The main spike will now sit exactly at index `pre_taps`
        h_centered = np.concatenate((precursor, postcursor), axis=-1)

        # Optional: Apply a windowing function to smoothly taper the edges to 0
        # window = np.hanning(pre_taps + post_taps)
        # h_centered = h_centered * window

        return h_centered

