""" channel_refdisres.py
  simulate reflections, dispersion,resonance; AC coupling (distortion) followed by NEXT,AWGN


     I) ===   Raw Channel (base dispertion + two delayed reflections + stub resonance) ===

- Base Dispertion: fundamental low-pass nature of the channel (skin effect and
    dielectric loss).  Implemented as the sum of 2 exponentials

- Two Reflections: (impedance mismatches)   1. short-delayed Tx Package/BGA breakout.
                                            2. Connector.
- Stub Resonance: Narrowband ringing/notch (Via stubs)

    II) === AC coupling ===

- Derivative Distortion: high pass characteristic; AC coupling capacitors, parasitic
    gap coupling.


    Application (in order):
- Tx scaling (simulate insertion loss)
- Channel Convolution
- NEXT
- AWGN

"""

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import skdim
import time


class ChannelRefDisRes:
    def __init__(self, config, B=1):
        """
        Matrix-vectorized LTI Channel Simulator for generating dynamic batches
        of channel realizations based on uniform range parameters.

        :param config: Dictionary parsed from the YAML configuration file
        :param B: Batch size (number of independent channel realizations)
        """
        self.config = config
        self.B = B

        # System parameters
        self.baud_rate = float(config['system']['baud_rate'])
        self.os_factor = int(config['system']['os_factor'])
        self.batch_size = int(config['system']['batch_size'])

        self.fs = self.baud_rate * self.os_factor
        self.dt = 1 / self.fs
        self.ui_sec = 1.0 / self.baud_rate

    def generate_batch(self):
        """
        Generate one main batch of impulse responses (h_a) and a second one (h_p), guaranteed
        small statistical departure from h_a

        """
        c = self.config['channel']

        draw = lambda key: np.random.uniform(c[f"{key}_min"], c[f"{key}_max"], size=self.B)

        # These are values which apply even if the channel is "ideal"

        self.insertion_loss_db = draw("insertion_loss_db")
        self.next_isolation_db = draw("insertion_loss_db")
        self.num_aggressors = draw("num_aggressors")
        self.snr_db = draw("snr_db")

        if c['mode'] == 'ideal':
            # Ideal Dirac Delta impulse response, properly batched as (B, 1)
            h_a = np.ones((self.batch_size, 1), dtype=np.complex64)
            h_p = np.ones((self.batch_size, 1), dtype=np.complex64)
            return h_a, h_p

        # Base Parameters (Anchor)
        tau_rise_a = draw("tau_rise")
        tau_fall_a = draw("tau_fall")
        stub_w = draw("stub_weight")
        f_res = draw("f_res_ghz") * 1e9
        deriv_w = draw("deriv_weight")

        # Poisson Cascaded Multipath (Anchor)
        lam = c.get("boundary_count_lambda", 2.5)
        N = np.random.poisson(lam, size=self.B)
        max_N = max(1, np.max(N))  # Ensure at least 1 column for vectorized math
        mask = np.arange(max_N)[None, :] < N[:, None]  # Boolean mask [B, max_N]

        # Draw gammas and delays
        g_sig = np.random.uniform(c["gamma_sigma_min"], c["gamma_sigma_max"], size=self.B)
        gamma_a = np.random.normal(c["gamma_mu"], g_sig[:, None], size=(self.B, max_N)) * mask

        # Draw absolute delays within the bounds
        delays_ui = np.random.uniform(c["absolute_delay_ui_min"], c["absolute_delay_ui_max"], size=(self.B, max_N))
        # Sort them so they happen chronologically (important for physics logic later if needed)
        delays_ui = np.sort(delays_ui, axis=1)
        tau_a = delays_ui * self.ui_sec * mask

        # Apply Micro-Drifts (Positive Pair)
        # Thermal smearing
        tau_rise_p = tau_rise_a * np.random.uniform(1.0 - c["drift_tau_pct"], 1.0 + c["drift_tau_pct"], size=self.B)
        tau_fall_p = tau_fall_a * np.random.uniform(1.0 - c["drift_tau_pct"], 1.0 + c["drift_tau_pct"], size=self.B)

        # Impedance ripple and Jitter
        gamma_p = gamma_a + np.random.normal(0, c["drift_gamma_sigma"], size=(self.B, max_N)) * mask
        tau_p = tau_a + np.random.uniform(-c["drift_delay_ui"], c["drift_delay_ui"],
                                          size=(self.B, max_N)) * self.ui_sec * mask

        # Synthesize Waveforms
        h_a = self._synthesize(tau_rise_a, tau_fall_a, gamma_a, tau_a, stub_w, f_res, deriv_w)
        h_p = self._synthesize(tau_rise_p, tau_fall_p, gamma_p, tau_p, stub_w, f_res, deriv_w)

        return h_a, h_p

    def _synthesize(self, tau_rise, tau_fall, gamma, tau, stub_w, f_res, deriv_w):
        """Transforms physical parameters into the 1D LTI waveform tensor."""
        max_initial_samples = 1500
        t = np.arange(0, max_initial_samples) * self.dt
        t_grid = t[np.newaxis, :]  # Shape [1, 1500]

        t_r = tau_rise[:, np.newaxis]
        t_f = tau_fall[:, np.newaxis]

        # Base Line-of-Sight Pulse
        h_base = np.exp(-t_grid / t_f) - np.exp(-t_grid / t_r)
        h_base = np.where(h_base > 0, h_base, 0.0)
        h_raw = np.copy(h_base)

        # Superimpose all valid reflections
        _, max_N = gamma.shape
        echo_attenuation_factor = float(self.config["channel"]["echo_attenuation_factor"])
        bounce_dispersion_max = float(self.config["channel"]["bounce_dispersion_max"])
        for i in range(max_N):
            g_i = gamma[:, i:i + 1]
            tau_i = tau[:, i:i + 1]

            t_shifted = t_grid - tau_i
            t_safe = np.maximum(t_shifted, 0.0)

            # --- APPLY DISTANCE-BASED PHYSICS ---
            # 1. Dispersion: The further it traveled (tau_i), the wider it gets
            # Max delay is approx 48 * dt. Scale the rise/fall times.
            max_delay_sec = 48 * self.ui_sec
            dispersion_ratio = 1.0 + (tau_i / max_delay_sec) * (bounce_dispersion_max - 1.0)
            t_r_echo = t_r * dispersion_ratio
            t_f_echo = t_f * dispersion_ratio

            # 2. Attenuation: The further it traveled, the more copper loss it suffered
            # Exponential decay based on distance
            atten_factor = echo_attenuation_factor
            distance_loss = np.exp(-tau_i * atten_factor / max_delay_sec)

            # Synthesize the physical echo
            h_echo = np.exp(-t_safe / t_f_echo) - np.exp(-t_safe / t_r_echo)
            h_echo = np.where(t_shifted > 0, g_i * h_echo * distance_loss, 0.0)

            h_raw += h_echo

        # Add Stub Resonance
        damping = 1.5e10
        h_stub = stub_w[:, None] * np.sin(2 * np.pi * f_res[:, None] * t_grid) * np.exp(-t_grid * damping)
        h_raw += h_stub

        # Add Derivative Peaking
        h_diff = np.diff(h_raw, axis=-1)
        h_diff = np.insert(h_diff, 0, 0.0, axis=-1)
        h_final = h_raw + (deriv_w[:, None] * h_diff)

        # Dynamic Energy Truncation (same as your original logic)
        energy = h_final ** 2
        cum_energy = np.cumsum(energy, axis=-1)
        energy_threshold = 0.99 * cum_energy[:, -1:]
        required_lengths = np.argmax(cum_energy >= energy_threshold, axis=-1) + 1

        tail_margin = int(20 * self.os_factor)
        max_batch_len = min(int(np.max(required_lengths)) + tail_margin, max_initial_samples)
        h_clipped = h_final[:, :max_batch_len]

        # DC Normalization
        row_sums = np.sum(h_clipped, axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        h_tensor = (h_clipped / row_sums).astype(np.float32)

        return h_tensor

    def estimate_h_dimension(self, X):
        # # 1. Load config
        # with open(config_path, 'r') as f:
        #     config = yaml.safe_load(f)
        #
        # print(f"-- Generating {num_samples} channel realizations...")
        # # 2. Generate a massive static batch
        # simulator = ChannelRefDisRes(config, B=num_samples)
        #
        # # We only care about the anchor (h_a) for macroscopic ID estimation
        # h_a, _ = simulator.generate_batch()
        #
        # # h_a is shape [10000, 400]. skdim expects [N_samples, N_features]
        # X = h_a
        print(f"-- Dataset shape: {X.shape}. Starting ID Estimation...\n")

        # 3. Run TwoNN (Fast Baseline)
        t0 = time.time()
        twonn = skdim.id.TwoNN().fit(X)
        print(f"TwoNN Estimated ID: {twonn.dimension_:.2f} (Took {time.time() - t0:.2f}s)")

        # 4. Run MLE (Localized Baseline)
        t0 = time.time()
        mle = skdim.id.MLE().fit(X)
        print(f"MLE Estimated ID:   {mle.dimension_:.2f} (Took {time.time() - t0:.2f}s)")

        # 5. Run DANCo (Heavy, highly accurate)
        # DANCo can take a while on 10,000 samples of 400 dims.
        t0 = time.time()
        danco = skdim.id.DANCo().fit(X)
        print(f"DANCo Estimated ID: {danco.dimension_:.2f} (Took {time.time() - t0:.2f}s)")

    def process(self, tx_analog_batch, h):
        """
        Processes a batch of complex Tx waveforms through the channel pipeline.
        Optimized for JAX/XLA via a single, full-frame linear FFT convolution.

        :param tx_analog_batch: 2D complex64 numpy array of shape (B, N_samples)
        :param h: 2D float32 numpy array representing the channel impulse response
        :return: 2D complex64 numpy array of the final Rx waveform
        """
        B, N_samples = tx_analog_batch.shape
        M = h.shape[1]

        # --- broadband attenuation
        linear_loss = 10 ** (-self.insertion_loss_db[:, np.newaxis] / 20.0)
        rx_attenuated = tx_analog_batch * linear_loss

        # --- full-frame fft convolution (Complex Signal * Real Channel)
        # To perform true linear convolution via FFT without circular wrap-around
        # (time aliasing), the FFT size must be exactly N + M - 1.
        full_len = N_samples + M - 1
        N_fft = 2 ** int(np.ceil(np.log2(full_len)))

        # Compute batched FFTs. 'rx_attenuated' is complex, so X is complex.
        X = np.fft.fft(rx_attenuated, n=N_fft, axis=-1)
        H = np.fft.fft(h, n=N_fft, axis=-1)

        # Multiply in frequency domain
        Y = X * H

        # IFFT back to time domain. Since Y is complex, rx_conv_full is complex.
        rx_conv_full = np.fft.ifft(Y, n=N_fft, axis=-1)

        # Truncate the mathematical tail to maintain the exact input array size
        rx_conv = rx_conv_full[:, :N_samples].astype(np.float32)

        # near-end crosstalk (NEXT) - Complex Baseband Aggressors
        num_symbols = (N_samples // self.os_factor) + 2
        crosstalk_noise = np.zeros_like(rx_conv, dtype=np.complex64)
        next_linear_gain = 10 ** (-self.next_isolation_db[:, np.newaxis] / 20.0)

        max_aggressors = int(np.max(self.num_aggressors))

        for i in range(max_aggressors):
            active_mask = (self.num_aggressors > i)[:, np.newaxis].astype(np.float32)

            # Generate complex QAM aggressor symbols
            agg_i = np.random.choice([-3.0, -1.0, 1.0, 3.0], size=(B, num_symbols))
            agg_q = np.random.choice([-3.0, -1.0, 1.0, 3.0], size=(B, num_symbols))
            agg_syms = agg_i + 1j * agg_q

            # Upsample
            agg_up = np.repeat(agg_syms, self.os_factor, axis=-1)[:, :N_samples]

            # Capacitive coupling derivative of the complex aggressor
            agg_diff = np.diff(agg_up, axis=-1)
            agg_diff = np.insert(agg_diff, 0, 0.0j, axis=-1)

            crosstalk_noise += (agg_diff * next_linear_gain * active_mask)

        rx_with_xtalk = rx_conv + crosstalk_noise

        # add AWGN (Circularly Symmetric Complex Thermal Noise)
        # np.var on a complex array correctly computes the variance of the magnitude
        sig_power = np.var(rx_conv, axis=-1, keepdims=True)
        snr_linear = 10 ** (self.snr_db[:, np.newaxis] / 10.0)

        # Total required noise power
        total_noise_power = sig_power / snr_linear

        # Generate independent Gaussian noise for I and Q, splitting the power in half
        awgn_real = np.random.normal(0.0, 1.0, size=rx_with_xtalk.shape)
        awgn_imag = np.random.normal(0.0, 1.0, size=rx_with_xtalk.shape)
        awgn = ((awgn_real + 1j * awgn_imag) * np.sqrt(total_noise_power / 2.0)).astype(np.complex64)

        rx_final = rx_with_xtalk + awgn

        return rx_final.astype(np.complex64)
