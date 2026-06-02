import numpy as np
from utils.fft_convolve_same import fft_convolve_same
from utils.bessel import generate_bessel_taps


def get_wifi_qam_lut(modulation='QAM16'):
    """Returns the 1D Gray-coded mapping and normalization factor for Wi-Fi standard QAM."""
    if modulation == 'QAM4':
        # 1 bit per channel: 0->-1, 1->+1
        lut = np.array([-1, 1], dtype=float)
        k_mod = 1.0 / np.sqrt(2)
        bits_per_ch = 1
    elif modulation == 'QAM16':
        # 2 bits per channel: 00->-3, 01->-1, 11->1, 10->3
        lut = np.array([-3, -1, 3, 1], dtype=float)
        k_mod = 1.0 / np.sqrt(10)
        bits_per_ch = 2
    elif modulation == 'QAM64':
        # 3 bits per channel: Wi-Fi standard 8-level mapping
        lut = np.array([-7, -5, -1, -3, 7, 5, 1, 3], dtype=float)
        k_mod = 1.0 / np.sqrt(42)
        bits_per_ch = 3
    else:
        raise ValueError("Unsupported modulation")
    return lut, k_mod, bits_per_ch


def generate_walsh_hadamard_am(batch_size, length=256, lane_idx=0):
    """Generates a 256-symbol QPSK Alignment Marker masked by a Walsh-Hadamard sequence."""
    # Generate a robust QPSK pseudo-random base sequence
    np.random.seed(42)  # Fixed seed so the marker is deterministic across runs
    base_i = np.random.choice([-1.0, 1.0], size=(1, length))
    base_q = np.random.choice([-1.0, 1.0], size=(1, length))

    # Simple 4x4 Walsh-Hadamard matrix rows for lane isolation
    wh_matrix = [
        np.ones(length),  # Lane 0: ++++++++++++++++
        np.tile([1, -1], length // 2),  # Lane 1: +-+-+-+-+-+-+-+-
        np.tile([1, 1, -1, -1], length // 4),  # Lane 2: ++--++--++--++--
        np.tile([1, -1, -1, 1], length // 4)  # Lane 3: +--++--++--++--+
    ]
    wh_seq = wh_matrix[lane_idx % 4]

    # Apply WH mask and broadcast to batch size
    am_i = np.tile(base_i * wh_seq, (batch_size, 1))
    am_q = np.tile(base_q * wh_seq, (batch_size, 1))
    return am_i, am_q


class PhysicalTransmitter:
    def __init__(self, config):
        self.config = config
        self.modulation = config['system'].get('modulation', 'QAM16')
        self.baud_rate = float(config['system']['baud_rate'])
        self.os_factor = int(config['system']['os_factor'])  # Analog OS (e.g., 8)
        self.T_ana = 1.0 / (self.baud_rate * self.os_factor)

        # --- PARSE YAML TX SETTINGS (with safe fallbacks) ---
        tx = config.get('tx', {})

        # 1. DAC
        dac_cfg = tx.get('dac', {})
        self.enob = dac_cfg.get('physical_bits', 8)
        self.clip_papr = np.random.uniform(dac_cfg.get('clip_papr_db_min', 5.5),
                                           dac_cfg.get('clip_papr_db_max', 9.0))
        self.inl_sigma = dac_cfg.get('inl_sigma_max', 1.5)
        self.dnl_sigma = dac_cfg.get('dnl_sigma_max', 0.8)

        # 2. Clocking
        clock_cfg = tx.get('clocking', {})
        self.jitter_rms = np.random.uniform(clock_cfg.get('random_jitter_rms_fs_min', 50.0),
                                            clock_cfg.get('random_jitter_rms_fs_max', 350.0)) * 1e-15

        # 3. IQ Imbalance
        iq_cfg = tx.get('iq_imbalance', {})
        self.iq_gain_db = np.random.normal(iq_cfg.get('gain_db_mu', 0.0),
                                           iq_cfg.get('gain_db_sigma_max', 0.8))
        self.iq_phase_deg = np.random.normal(iq_cfg.get('phase_deg_mu', 0.0),
                                             iq_cfg.get('phase_deg_sigma_max', 4.0))

        # 4. Modulator
        mod_cfg = tx.get('modulator', {})
        self.vpp_vpi = np.random.uniform(mod_cfg.get('vpp_to_vpi_ratio_min', 0.4),
                                         mod_cfg.get('vpp_to_vpi_ratio_max', 0.9))
        self.bias_drift = np.random.uniform(mod_cfg.get('bias_drift_pct_min', -8.0),
                                            mod_cfg.get('bias_drift_pct_max', 8.0)) / 100.0
        er_db = np.random.uniform(mod_cfg.get('extinction_ratio_db_min', 16.0),
                                  mod_cfg.get('extinction_ratio_db_max', 28.0))
        self.er_amp = 10 ** (-er_db / 20.0)

        # 5. RF Driver / MPM
        rf_cfg = tx.get('rf_driver', {})
        self.mpm_k = rf_cfg.get('poly_order_k', 5)
        self.mpm_m = rf_cfg.get('memory_depth_m', 4)
        sigma_nl = rf_cfg.get('non_linear_coefficients_sigma', 0.15)

        # Generate MPM Coefficient Matrix (Memory Depth x Odd Orders)
        self.orders = np.arange(1, self.mpm_k + 1, 2)
        self.mpm_coeffs = np.random.normal(0, sigma_nl, size=(self.mpm_m, len(self.orders)))
        self.mpm_coeffs[0, 0] = 1.0  # Main instantaneous linear tap must be normalized to ~1.0

        # --- PRE-CALCULATE HARDWARE TABLES & FILTERS ---
        self._generate_dac_grid()

        # Bessel Filter Cutoff (properly normalized against Simulation Nyquist)
        cutoff_ratio = 1.3 / self.os_factor  #  1 is Simulation Nyquist
        span = 64 # number of taps at simulation frequency
        self.tx_afe_taps = generate_bessel_taps(cutoff_ratio=cutoff_ratio, span=span)

    def _generate_dac_grid(self):
        """Generates the static INL/DNL voltage map for the DAC."""
        levels = 2 ** self.enob
        ideal_steps = np.linspace(-1, 1, levels)

        # DNL: Random step variations
        dnl = np.random.normal(0, self.dnl_sigma / levels, levels)
        dnl[0] = 0  # Anchor first step

        # INL: Cumulative sum of DNL, forced to match endpoints, minus DC tilt
        inl = np.cumsum(dnl)
        inl_correction = np.linspace(inl[0], inl[-1], levels)
        inl = inl - inl_correction

        # Add INL bow (low frequency warp)
        bow = np.sin(np.linspace(0, np.pi, levels)) * (self.inl_sigma / levels)

        self.dac_grid = ideal_steps + inl + bow

    def transmit_frame(self, batch_size, payload_bits):
        """Executes the full pipeline for a batched row tensor."""
        _, _, bits_per_ch = get_wifi_qam_lut(self.modulation)
        payload_symbols = payload_bits // bits_per_ch

        # Generate 256-symbol AM and Payload
        am_i, am_q = generate_walsh_hadamard_am(batch_size, length=256, lane_idx=0)
        # Apply standard Wi-Fi normalization to QPSK AM to match payload power
        am_i *= (1.0 / np.sqrt(2))
        am_q *= (1.0 / np.sqrt(2))

        pay_i, pay_q = self._step1_generate_payload(batch_size, payload_symbols)

        # Concatenate in the 100 Gbd digital domain
        I_sym = np.concatenate((am_i, pay_i), axis=1)
        Q_sym = np.concatenate((am_q, pay_q), axis=1)

        # 2. DAC Digital Physics (200 GSa/s)
        I_dac, Q_dac = self._step2_dac_digital_domain(I_sym, Q_sym)

        # 3. Electrical Analog Physics (800 GSa/s)
        I_rf, Q_rf = self._step3_analog_domain(I_dac, Q_dac)

        # 4. Electro-Optic Physics (800 GSa/s Complex Baseband)
        tx_optical_field = self._step4_modulator_physics(I_rf, Q_rf)

        return tx_optical_field

    def _step1_generate_payload(self, batch_size, payload_symbols):
        """Generates random payload and maps to discrete QAM voltage levels."""
        lut, k_mod, bits_per_ch = get_wifi_qam_lut(self.modulation)

        i_idx = np.random.randint(0, 2 ** bits_per_ch, size=(batch_size, payload_symbols))
        q_idx = np.random.randint(0, 2 ** bits_per_ch, size=(batch_size, payload_symbols))

        I_payload = lut[i_idx] * k_mod
        Q_payload = lut[q_idx] * k_mod
        return I_payload, Q_payload

    def _step2_dac_digital_domain(self, I_sym, Q_sym):
        """100Gbd to 200GSa/s (2x OS), OQAM Shift, and DAC Quantization."""
        # A. Upsample to 2x (200 GSa/s)
        I_2x = np.repeat(I_sym, 2, axis=1)
        Q_2x = np.repeat(Q_sym, 2, axis=1)

        # B. Offset QAM (Shift Q by exactly 1 sample at 2x rate = T/2)
        padding = np.zeros((I_2x.shape[0], 1))
        I_oqam = np.concatenate((I_2x, padding), axis=1)
        Q_oqam = np.concatenate((padding, Q_2x), axis=1)

        # C. PAPR Clipping
        clip_linear = 10 ** (self.clip_papr / 20.0)

        # D. Quantization (Nearest-neighbor mapping to pre-calculated INL/DNL grid)
        def quantize_to_grid(x):
            x_norm = np.clip(x / clip_linear, -1, 1)
            # Find closest index in the dac_grid using broadcasting
            idx = np.abs(x_norm[..., np.newaxis] - self.dac_grid).argmin(axis=-1)
            return self.dac_grid[idx] * clip_linear

        I_quantized = quantize_to_grid(I_oqam)
        Q_quantized = quantize_to_grid(Q_oqam)

        return I_quantized, Q_quantized

    def _step3_analog_domain(self, I_2x, Q_2x):
        """200GSa/s to 800GSa/s (4x OS), Filtering, and Volterra PA Non-linearities."""
        # A. Continuous Time Projection (800 GSa/s)
        analog_os = self.os_factor // 2
        I_8x = np.repeat(I_2x, analog_os, axis=1).astype(np.float32)
        Q_8x = np.repeat(Q_2x, analog_os, axis=1).astype(np.float32)

        # B. Jitter Injection (Taylor Series Derivative Approximation)
        dt_jitter_I = np.random.normal(0, self.jitter_rms, I_8x.shape)
        dt_jitter_Q = np.random.normal(0, self.jitter_rms, Q_8x.shape)

        I_8x += np.gradient(I_8x, axis=1) * (dt_jitter_I / self.T_ana)
        Q_8x += np.gradient(Q_8x, axis=1) * (dt_jitter_Q / self.T_ana)

        # C. Tx AFE Linear Low-Pass Filtering
        I_filt = fft_convolve_same(I_8x, self.tx_afe_taps)
        Q_filt = fft_convolve_same(Q_8x, self.tx_afe_taps)

        # D. RF Driver Memory Polynomial (MPM)
        I_nl = np.zeros_like(I_filt)
        Q_nl = np.zeros_like(Q_filt)

        # Generalized Volterra loop using YAML parameters
        for m in range(self.mpm_m):
            I_del = np.roll(I_filt, shift=m, axis=1)
            Q_del = np.roll(Q_filt, shift=m, axis=1)

            # Clean wrap-around edges
            if m > 0:
                I_del[:, :m] = I_filt[:, 0:1]
                Q_del[:, :m] = Q_filt[:, 0:1]

            mag_sq_del = I_del ** 2 + Q_del ** 2

            for idx, k in enumerate(self.orders):
                c = self.mpm_coeffs[m, idx]
                power = (k - 1) // 2
                I_nl += c * I_del * (mag_sq_del ** power)
                Q_nl += c * Q_del * (mag_sq_del ** power)

        return I_nl, Q_nl

    def _step4_modulator_physics(self, I_nl, Q_nl):
        """Apply Analog I/Q Imbalances and Electro-Optic Modulator sine curve."""
        # A. I/Q Imbalance (Gain and Phase applied in continuous domain)
        gain_lin = 10 ** (self.iq_gain_db / 20.0)
        phase_rad = np.radians(self.iq_phase_deg)

        # Gain mismatch (I is slightly larger, Q is slightly smaller)
        I_imb = I_nl * np.sqrt(gain_lin)

        # Phase mismatch (Q bleeds into I)
        Q_imb = (Q_nl * np.cos(phase_rad) - I_nl * np.sin(phase_rad)) / np.sqrt(gain_lin)

        # B. Mach-Zehnder Modulator Transfer Function
        # Vpi normalized sine response: sin(V * Vpp/Vpi * pi/2 + bias_drift)
        theta_I = I_imb * self.vpp_vpi * (np.pi / 2) + (self.bias_drift * np.pi)
        theta_Q = Q_imb * self.vpp_vpi * (np.pi / 2) + (self.bias_drift * np.pi)

        # Add Extinction Ratio floor (finite null depth)
        I_opt = np.sin(theta_I) + self.er_amp
        Q_opt = np.sin(theta_Q) + self.er_amp

        # C. Return strictly as complex64 for TPU/JAX alignment
        return (I_opt + 1j * Q_opt).astype(np.complex64)
