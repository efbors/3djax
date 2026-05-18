import numpy as np


class DspReceiver:
    def __init__(self, num_taps=21, mu=5e-4):
        self.num_taps = num_taps
        self.mu = mu
        self.weights = np.zeros(num_taps)
        # Center the main cursor. For 21 taps, index 10 is the middle.
        self.center_tap = num_taps // 2
        self.weights[self.center_tap] = 1.0

    def apply_ffe_lms_genie(self, rx_analog, tx_symbols, os_factor, h_channel):
        """Runs a T-spaced FFE using the known TX symbols to train."""

        # 1. Genie Timing: Find the peak of the channel to get the optimal phase
        peak_idx = np.argmax(h_channel)
        best_phase = peak_idx % os_factor
        delay_ui = peak_idx // os_factor

        # Downsample to 1x Baud Rate (T-spaced)
        rx_baud = rx_analog[best_phase::os_factor]

        eq_out_history = np.zeros_like(rx_baud)
        error_history = np.zeros_like(rx_baud)

        # We need to align the Rx data with the Tx symbols.
        # The total delay = (Channel Delay) + (FFE Center Tap Delay)
        total_delay = delay_ui + self.center_tap

        print(f"Starting LMS Training... Total Symbol Delay: {total_delay}")

        # Run the LMS Loop
        # We start at num_taps to ensure we have a full history buffer
        for i in range(self.num_taps, len(rx_baud)):
            # Extract the history buffer (newest sample first)
            # e.g., rx_baud[i], rx_baud[i-1], ..., rx_baud[i-20]
            eq_in = rx_baud[i: i - self.num_taps: -1]

            # 1. Filter: Dot product of weights and input
            eq_out = np.dot(self.weights, eq_in)
            eq_out_history[i] = eq_out

            # 2. Genie Error: Compare to the known transmitted symbol
            # Make sure we don't read out of bounds of the tx_symbols array
            tx_idx = i - total_delay
            if 0 <= tx_idx < len(tx_symbols):
                ref = tx_symbols[tx_idx]
                error = ref - eq_out
                error_history[i] = error

                # 3. LMS Update: W = W + mu * error * input
                self.weights += self.mu * error * eq_in

        return eq_out_history, error_history
