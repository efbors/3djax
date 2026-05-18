import numpy as np


class AGC:
    def __init__(self, config):
        agc_cfg = config['rx']['agc']
        self.target_power = float(agc_cfg['agc_target_power'])
        self.alpha_hot = float(agc_cfg['agc_alpha_hot'])
        self.alpha_cold = float(agc_cfg['agc_alpha_cold'])
        self.squelch_threshold = float(agc_cfg['squelch_threshold'])

        # State variables carried across blocks
        self.current_vga_gain = float(agc_cfg['current_vga_gain'])
        self.is_locked = False

        # System parameters
        self.window_size = int(config['system'].get('agc_window_size'))

    def set_alpha(self, alpha):
        self.current_alpha = alpha

    def process(self, analog_block):
        """
        Takes a 1D analog block, reshapes it into non-overlapping windows,
        and updates the stateful VGA gain once per window.
        """
        block_len = len(analog_block)

        # Guard: Ensure the block divides perfectly by the window size
        assert block_len % self.window_size == 0, f"Block length {block_len} not divisible by window size {self.window_size}"
        num_windows = block_len // self.window_size

        # Reshape into (num_windows, window_size)
        windows = analog_block.reshape(num_windows, self.window_size)

        # Pre-allocate the output arrays
        leveled_windows = np.zeros_like(windows)
        gain_trajectory = np.zeros(num_windows, dtype=np.float32)

        # The Windowed Execution Loop (Future jax.lax.scan target)
        for w_idx in range(num_windows):
            window_data = windows[w_idx]

            # Calculate power E[x^2]
            pwr = np.mean(window_data ** 2)

            if pwr > (self.squelch_threshold ** 2):
                # Calculate the mathematically ideal gain for this specific window
                safe_pwr = max(pwr, 1e-12)
                ideal_gain = np.sqrt(self.target_power / safe_pwr)

                # Gear Shifting: Hot vs Cold convergence
                # (For now, a simple threshold to switch gears. In hardware, 
                # this might be controlled by a timer or state machine)
                error_ratio = abs(ideal_gain - self.current_vga_gain) / self.current_vga_gain
                current_alpha = self.alpha_hot if error_ratio > 0.1 else self.alpha_cold

                # Stateful IIR update
                self.current_vga_gain = (1.0 - current_alpha) * self.current_vga_gain + (current_alpha * ideal_gain)

            # Apply the active gain uniformly across this 256-sample slice
            leveled_windows[w_idx] = window_data * self.current_vga_gain

            # Record state for the plot
            gain_trajectory[w_idx] = self.current_vga_gain

        # Flatten back to a 1D contiguous block
        leveled_block_out = leveled_windows.ravel()

        return leveled_block_out, gain_trajectory
