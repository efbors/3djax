import numpy as np
import matplotlib.pyplot as plt


class AGC:
    """ Block AGC

    """

    def __init__(self, config):
        agc_cfg = config['rx']['agc']
        self.target_power = float(agc_cfg['agc_target_power'])
        self.alpha_hot = float(agc_cfg['agc_alpha_hot'])
        self.alpha_cold = float(agc_cfg['agc_alpha_cold'])
        self.ADC_input_max = config['rx']['ADC_input_max']
        self.window_size = config['rx']['agc']['agc_window_size']

        # state
        self.alpha = 0.
        self.current_vga_gain = config['rx']['agc']['current_vga_gain']

    def set_alpha(self, alpha):
        self.alpha = alpha
    def get_current_gain(self):
            return self.current_vga_gain

    def process(self, time_series_in):
        """
        Standard Closed-Loop Block AGC.
        mu acts as the loop bandwidth / step size parameter.
        """
        block_len = len(time_series_in)
        num_windows = block_len // self.window_size
        windows_in = time_series_in[:num_windows * self.window_size].reshape(num_windows, self.window_size)

        time_series_out = np.empty_like(windows_in)
        gain_hist = np.empty(block_len, dtype=np.float32)
        for w_idx in range(num_windows):
            window_data = windows_in[w_idx]

            # 1. Apply current gain FIRST (Feedback architecture)
            agc_out_blk = window_data * self.current_vga_gain

            # 2. Estimate OUTPUT power
            pwr_out = np.mean(agc_out_blk ** 2)

            # 3. Calculate Error
            error = self.target_power - pwr_out

            # 4. Update Gain via loop filter (integrator)
            self.current_vga_gain = self.current_vga_gain + (self.alpha * error)

            # (Optional but recommended) Clamp the gain so it doesn't drop below 0 or explode
            self.current_vga_gain = np.clip(self.current_vga_gain, 1e-4, 100.0)

            # Store the clipped output and state
            time_series_out[w_idx] = np.clip(agc_out_blk, -self.ADC_input_max, self.ADC_input_max)

            start_idx = w_idx * self.window_size
            end_idx = start_idx + self.window_size
            gain_hist[start_idx:end_idx] = self.current_vga_gain

        time_series_out = time_series_out.ravel()

        return time_series_out, gain_hist

    # def process(self, time_series_in, alpha):
    #     """
    #     Takes a 1D time series at Fdig, reshapes it into non-overlapping windows,
    #     and
    #     """
    #
    #     block_len = len(time_series_in)
    #     assert block_len % self.window_size == 0, f"Block length {block_len} not divisible by window size {self.window_size}"
    #     num_windows = block_len // self.window_size
    #
    #     # Reshape into (num_windows, window_size)
    #     windows_in = time_series_in[:num_windows * block_len].reshape(num_windows, self.window_size)
    #     time_series_out = np.empty_like(windows_in)
    #     gain_hist = np.empty(num_windows, dtype=np.float32)
    #
    #     # The Windowed Execution Loop (Future jax.lax.scan target)
    #     for w_idx in range(num_windows):
    #         window_data = windows_in[w_idx]
    #
    #         pwr = np.mean(window_data ** 2)  # Calculate power E[x^2]
    #         pwr_sat = max(pwr, 1e-12)
    #         ideal_gain = np.sqrt(self.target_power / pwr_sat)
    #
    #         # Stateful IIR update
    #         self.current_vga_gain = (1.0 - alpha) * self.current_vga_gain + (alpha * ideal_gain)
    #
    #         agc_out_blk = window_data * self.current_vga_gain
    #         # Apply ADC saturation / hard clipping
    #         time_series_out[w_idx] = np.clip(agc_out_blk, -self.ADC_input_max, self.ADC_input_max)
    #
    #         # Record state for the plot
    #         gain_hist[w_idx] = self.current_vga_gain
    #
    #     time_series_out = time_series_out.ravel()
    #     return time_series_out, gain_hist

    def detect_sync_sequence(self, rx_analog, os_factor, sync_block):
        """
        Pure Digital Matched Filter.
        Uses a normalized rectangular template of the sync block.
        """

        # Upsample the raw PAM4 symbols (Rectangular pulses / ZOH)
        template = np.repeat(sync_block, os_factor).astype(np.float32)

        # Zero-mean and Normalize to unit energy
        template -= np.mean(template)
        norm_factor = np.linalg.norm(template)
        if norm_factor > 0:
            template /= norm_factor

        # Perform the cross-correlation
        correlation = np.correlate(rx_analog, template, mode='valid')

        # Find the exact alignment index
        sync_idx = int(np.argmax(np.abs(correlation)))
        sync_idx += len(template)
        if False:
            #  Plot the result
            import matplotlib.pyplot as plt
            plt.figure(figsize=(18, 8))

            plt.plot(np.abs(correlation), 'g', label='Matched Filter Output')
            plt.axvline(sync_idx, color='magenta', linestyle='--', linewidth=2, label=f'Sync Lock @ Index {sync_idx}')

            plt.title("Normalized Matched Filter: Optimal Cross-Correlation")
            plt.xlabel("Analog Indices")
            plt.ylabel("Correlation Magnitude")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show(block=False)

        return sync_idx, correlation
