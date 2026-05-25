""" channel_200G_hard.py
 - 1. at the simplest this simulates a Linear-Time-Invariant channel.
        (tau_fall_dynamic_en:False and
 - 2. simulate Thermal Dielectrich Shift. This is simulated by adding
     a slow, low-frequency modifier;  essentially increase dynamically
     tau_fall below. This expends/contracts the channel tail and exercises
     the FFE adaptation process.

 - 3. Mechanical Micro-vibrations;  introduce amplitude modulation of
    conn_weight and idx_conn (the connector reflection index);
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


class Channel200GHard:
    def __init__(self, config):
        self.config = config
        self.baud_rate = float(config['system']['baud_rate'])
        self.os_factor = int(config['system']['os_factor'])
        self.fs = self.baud_rate * self.os_factor
        self.dt = 1 / self.fs
        self.snr_db = config['channel']['snr_db']
        self.insertion_loss_db = config['channel']['insertion_loss_db']
        self.next_isolation_db = config['channel']['next_isolation_db']
        self.num_aggressors = config['channel']['num_aggressors']

        # --- Dynamic Channel State & Knobs ---
        self.current_time = 0.0  # Master timekeeper for the channel

        self.tau_fall_dynamic_en = config['channel'].get('tau_fall_dynamic_en', False)
        self.tau_fall_var_percent = float(config['channel'].get('tau_fall_var_percent', 0.0))
        self.tau_fall_period = float(config['channel'].get('tau_fall_period', 1.0))
        self.nominal_tau_fall = float(config['channel']['tau_fall'])

        self.conn_dynamic_en = config['channel'].get('conn_dynamic_en', False)
        self.conn_weight_var = float(config['channel'].get('conn_weight_var', 0.0))

        self.conn_period = float(config['channel'].get('conn_period', 1.0))
        self.nominal_conn_weight = float(config['channel']['conn_weight'])

        # Build the impulse response upon initialization
        self.h = self._gen_impulse_response()

        # --- Overlap-Save FFT Setup ---
        self.L = int(config['channel'].get('block_processing_size', 32768))
        self.M = len(self.h)  # Expected to be 500 based on t = np.arange(0, 500)
        self.N_fft = self.L + self.M - 1


    def process(self, tx_analog):
        """
                Passes the analog waveform through the channel, applying:
                1. Broadband Insertion Loss
                2. Channel Dispersion
                3. NEXT (Near-End Crosstalk from adjacent lanes)
                4. AWGN
                5. Receiver PPM Frequency Offset
        """

        # Apply Broadband Attenuation
        # Voltage drops by a factor of 10^(-dB/20)
        # e.g., 30 dB loss -> signal is ~3.16% of its original amplitude
        linear_attenuation = 10 ** (-self.insertion_loss_db / 20.0)
        tx_attenuated = tx_analog * linear_attenuation

        # --   Time-Variant Overlap-Save using Pure Numpy FFT
        if not (self.tau_fall_dynamic_en or self.conn_dynamic_en):
            # Fast path for static LTI channel
            rx_conv = np.convolve(tx_attenuated, self.h, mode='full')[:len(tx_attenuated)]
        else:
            original_len = len(tx_attenuated)

            # Pad the front with M - 1 zeros (historical state for block 0)
            padded_tx = np.pad(tx_attenuated, (self.M - 1, 0), mode='constant')

            # Pad the tail so the remaining data is a perfect multiple of L
            remainder = original_len % self.L
            if remainder != 0:
                pad_end = self.L - remainder
                padded_tx = np.pad(padded_tx, (0, pad_end), mode='constant')

            num_blocks = (len(padded_tx) - self.M + 1) // self.L

            # Master array for the concatenated output
            rx_conv_full = np.zeros(num_blocks * self.L, dtype=np.float32)

            for k in range(num_blocks):
                # Extract chunk of size N_fft
                start_idx = k * self.L
                end_idx = start_idx + self.N_fft
                chunk = padded_tx[start_idx:end_idx]

                # Generate dynamic impulse response for this specific moment
                self.h = self._gen_impulse_response()

                # FFT-based Circular Convolution
                H = np.fft.fft(self.h, n=self.N_fft)
                X = np.fft.fft(chunk, n=self.N_fft)
                Y = H * X
                y_chunk = np.real(np.fft.ifft(Y))

                # Overlap-Save: Discard the first M-1 aliased samples
                valid_samples = y_chunk[self.M - 1:]

                # Store exactly L clean samples in the master array
                out_start = k * self.L
                out_end = out_start + self.L
                rx_conv_full[out_start:out_end] = valid_samples

                # Advance the master clock by the duration of the valid block
                self.current_time += self.L * self.dt

            # Trim the final padded tail to perfectly match the exact input length
            rx_conv = rx_conv_full[:original_len]

        # Simulate Realistic NEXT (Near-End Crosstalk)
        crosstalk_noise = np.zeros_like(rx_conv)
        next_linear_gain = 10 ** (-self.next_isolation_db / 20.0)

        # We only need roughly the right number of symbols to cover the array
        num_agg_symbols = (len(rx_conv) // self.os_factor) + 2

        for i in range(self.num_aggressors):
            # Generate a raw, unshaped PAM4 aggressor signal
            agg_symbols = np.random.choice([-3.0, -1.0, 1.0, 3.0], size=num_agg_symbols)
            agg_up = np.repeat(agg_symbols, self.os_factor)

            # Apply Tx pulse shaping to the aggressor so the edges are realistic.
            agg_shaped = gaussian_filter1d(agg_up, sigma=1.0)
            # Trim to perfectly match our victim array length
            agg_shaped = agg_shaped[:len(rx_conv)]

            # The capacitive coupling (derivative) creates a smooth, band-limited bump
            agg_coupled = np.append(np.diff(agg_shaped), 0.0)

            # Add to the crosstalk pool
            crosstalk_noise += (agg_coupled * next_linear_gain)

        # Add the crosstalk interference to our victim lane
        rx_conv += crosstalk_noise

        # Add AWGN
        # Calculate SNR based on the *attenuated* signal power.
        # This properly models the thermal noise floor floor at the RX pads.
        signal_power = np.var(rx_conv)
        noise_power = signal_power / (10 ** (self.snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(rx_conv)).astype(np.float32)
        channel_out = rx_conv + noise
        return channel_out

    def _gen_impulse_response(self):
        """Creates a highly configurable DCI channel impulse response."""

        # Read knobs from config
        mode = self.config['channel'].get('response_mode', 'physical')
        pkg_weight = float(self.config['channel']['pkg_weight'])
        stub_weight = float(self.config['channel']['stub_weight'])
        tau_rise = self.config['channel']['tau_rise']

        # --- Calculate Dynamic tau_fall ---
        if self.tau_fall_dynamic_en:
            swing = self.nominal_tau_fall * (self.tau_fall_var_percent / 100.0)
            tau_fall = self.nominal_tau_fall + swing * np.sin(2 * np.pi * self.current_time / self.tau_fall_period)
        else:
            tau_fall = self.nominal_tau_fall

        # --- Calculate Dynamic conn_weight ---
        if self.conn_dynamic_en:
            conn_weight = self.nominal_conn_weight + self.conn_weight_var * np.sin(
                2 * np.pi * self.current_time / self.conn_period)
        else:
            conn_weight = self.nominal_conn_weight

        t = np.arange(0, 500) * self.dt  # 125 UI span

        # ---------------------------------------------------------
        # LEVEL 0: PERFECT PASS-THROUGH (Dirac Delta)
        # ---------------------------------------------------------
        if mode == 'ideal':
            h = np.zeros_like(t)
            h[0] = 1.0
            return h

        # ---------------------------------------------------------
        # LEVEL 1: BASE DISPERSION (Skin effect + Dielectric loss)

        # ---------------------------------------------------------
        """
        Double-Exponential Impulse Response.  base impulse response of a band-limited
        transmission line; models 1. causality , 2. rise time, 3. fall time (longer tail
        of the skin effect and dielectric loss
        """
        h_base = (np.exp(-t / tau_fall) - np.exp(-t / tau_rise))
        h_base = np.where(h_base > 0, h_base, 0)

        # ---------------------------------------------------------
        # LEVEL 2: REFLECTIONS & RINGING (Dialed via YAML)
        # ---------------------------------------------------------
        # Package Mismatch (Short echo)
        idx_pkg = int(2.5 * self.os_factor)
        h_pkg = np.zeros_like(t)
        h_pkg[idx_pkg:] = pkg_weight * h_base[:-idx_pkg]

        # Via Stub Resonance (High-frequency ringing)
        f_res, damping = 35e9, 1.5e10
        h_stub = stub_weight * np.sin(2 * np.pi * f_res * t) * np.exp(-t * damping)

        # Connector Reflection (Long echo)
        idx_conn = int(14.2 * self.os_factor)
        h_conn = np.zeros_like(t)
        h_conn[idx_conn:] = conn_weight * h_base[:-idx_conn]

        # Combine
        h = h_base + h_pkg + h_stub + h_conn

        # Normalize so DC gain is exactly 1.0
        h_norm = (h / np.sum(h)).astype(np.float32)

        return h_norm

    @staticmethod
    def plot_eye(rx_signal, os_factor, title="PAM4 Eye Diagram", delay_ui=0):
        import matplotlib.pyplot as plt
        """A high-performance 2D-histogram eye diagram plotter."""
        samples_per_trace = os_factor * 2
        trim_len = (len(rx_signal) - delay_ui * os_factor) // samples_per_trace * samples_per_trace
        if trim_len <= 0: return

        traces = rx_signal[delay_ui * os_factor: delay_ui * os_factor + trim_len]
        traces = traces.reshape(-1, samples_per_trace)
        time_ui = np.linspace(0, 2, samples_per_trace)

        plt.figure(figsize=(8, 6))
        for i in range(min(2000, len(traces))):
            plt.plot(time_ui, traces[i], color='blue', alpha=0.05)

        plt.title(title)
        plt.xlabel("Time (UI)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.show()
