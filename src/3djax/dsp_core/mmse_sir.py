import numpy as np
import scipy.linalg


class MmseSir:
    def __init__(self, config):
        self.config = config

    def _extract_centered_taps(self, h_full, pre_taps, post_taps):
        """
        Extracts the causal physical linear channel from the circular
        ZF IFFT output array, centering the main energy.
        """
        # Grab the wrapped precursor from the end of the array
        precursor = h_full[:, -pre_taps:]
        # Grab the main peak and postcursor from the start
        postcursor = h_full[:, :post_taps]

        # Stitch them into a single linear filter [Batch, L_h]
        return np.concatenate((precursor, postcursor), axis=-1)

    def calculate_weights(self, h_full_batch, snr_linear_batch, ffe_taps=64, target_taps=3, pre_taps=16, post_taps=48):
        """
        Calculates the MMSE-SIR FFE weights and the optimal Target ISI.

        :param h_full_batch: Circular channel estimate from ZF (Shape: B, 2003)
        :param snr_linear_batch: True linear SNR estimated from the ZF noise floor
        :param ffe_taps: The number of FFE filter weights (Nw)
        :param target_taps: The residual ISI length for the Sequence Estimator (Ng)
        """
        # Convert arrays to batched if running a single vector
        if h_full_batch.ndim == 1:
            h_full_batch = h_full_batch[np.newaxis, :]
            snr_linear_batch = np.array([snr_linear_batch])

        batch_size = h_full_batch.shape[0]

        # 1. Extract physical linear channel
        h_centered = self._extract_centered_taps(h_full_batch, pre_taps, post_taps)
        L_h = pre_taps + post_taps
        full_len = ffe_taps + L_h - 1

        # Output allocations
        w_batch = np.zeros((batch_size, ffe_taps), dtype=np.complex128)
        g_batch = np.zeros((batch_size, target_taps), dtype=np.complex128)
        delay_batch = np.zeros(batch_size, dtype=np.int32)

        for b in range(batch_size):
            h = h_centered[b]
            snr = snr_linear_batch[b]

            # 2. Build the Toeplitz channel convolution matrix H
            # Shape: (ffe_taps, ffe_taps + L_h - 1)
            H = np.zeros((ffe_taps, full_len), dtype=np.complex128)
            for i in range(ffe_taps):
                H[i, i: i + L_h] = h

            # 3. Autocorrelation R_yy
            # R_yy = H * H^H + (1/SNR) * I
            sigma_n2 = 1.0 / snr
            R_yy = H @ H.conj().T + sigma_n2 * np.eye(ffe_taps)

            # 4. Global Error Covariance Matrix R_xx_eff
            R_yy_inv = scipy.linalg.inv(R_yy)
            H_H = H.conj().T
            R_xx_eff = np.eye(full_len) - H_H @ R_yy_inv @ H

            # 5. Sweep delays to find the globally optimal target g_opt
            min_mse = float('inf')
            best_g = None
            best_delay = 0

            valid_delays = full_len - target_taps + 1
            for delta in range(valid_delays):
                # Extract the Ng x Ng submatrix on the diagonal
                R_sub = R_xx_eff[delta: delta + target_taps, delta: delta + target_taps]

                # Eigenvalue decomposition (eigh is optimized for Hermitian matrices)
                vals, vecs = np.linalg.eigh(R_sub)

                # The minimum eigenvalue is the mathematical Minimum Mean-Square Error
                mse = vals[0]
                if mse < min_mse:
                    min_mse = mse
                    best_g = vecs[:, 0]  # The eigenvector corresponding to min MSE
                    best_delay = delta

            # 6. Calculate optimal FFE weights (w) for the best delay
            d = np.zeros(full_len, dtype=np.complex128)
            d[best_delay: best_delay + target_taps] = best_g

            w_opt = R_yy_inv @ H @ d

            # 7. Phase and Gain Normalization (CRITICAL FOR QAM)
            # Find the peak tap and force it to 1.0 + 0j.
            # This preserves exact Euclidean distance for the Log-MAP LLRs.
            peak_idx = np.argmax(np.abs(best_g))
            phase_correction = best_g[peak_idx]

            best_g = best_g / phase_correction
            w_opt = w_opt / phase_correction

            w_batch[b] = w_opt
            g_batch[b] = best_g
            delay_batch[b] = best_delay

        return w_batch, g_batch, delay_batch