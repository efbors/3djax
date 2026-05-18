import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
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

        # Build the impulse response upon initialization
        self.h = self._gen_impulse_response()

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

        # Convolve with physical channel (Dispersion & Reflections)
        rx_conv = np.convolve(tx_attenuated, self.h, mode='full')[:len(tx_attenuated)]

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
        conn_weight = float(self.config['channel']['conn_weight'])
        tau_rise = self.config['channel']['tau_rise']
        tau_fall = self.config['channel']['tau_fall']

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
