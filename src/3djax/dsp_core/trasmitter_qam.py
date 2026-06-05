import numpy as np
from utils.fft_convolve_same import fft_convolve_same
from utils.bessel import generate_bessel_taps
from utils.qam_mapper import get_qam_lut
from utils.diagnostics import oqam_eye


def generate_walsh_hadamard_am(length, lane_idx=0):
    """Generates a 256-symbol QPSK Alignment Marker masked by a Walsh-Hadamard sequence."""
    # Generate a robust QPSK pseudo-random base sequence
    np.random.seed(42)  # Fixed seed so the marker is deterministic across runs
    base_i = np.random.choice([-1.0, 1.0], size=(1, length)).astype(np.float32)
    base_q = np.random.choice([-1.0, 1.0], size=(1, length)).astype(np.float32)

    # Simple 4x4 Walsh-Hadamard matrix rows for lane isolation
    wh_matrix = [
        np.ones(length),  # Lane 0: ++++++++++++++++
        np.tile([1, -1], length // 2),  # Lane 1: +-+-+-+-+-+-+-+-
        np.tile([1, 1, -1, -1], length // 4),  # Lane 2: ++--++--++--++--
        np.tile([1, -1, -1, 1], length // 4)  # Lane 3: +--++--++--++--+
    ]
    wh_seq = wh_matrix[lane_idx % 4]

    # Apply WH mask and broadcast to batch size
    am_i = base_i * wh_seq.astype(np.float32)
    am_q = base_q * wh_seq.astype(np.float32)
    am_ref = am_i + 1j * am_q
    return am_ref


class PhysicalTransmitter:
    def __init__(self, config):
        self.config = config
        self.modulation = config['system']['modulation']
        self.baud_rate = float(config['system']['baud_rate'])
        self.os_factor = int(config['system']['os_factor'])  # Analog OS (e.g., 8)
        self.T_ana = 1.0 / (self.baud_rate * self.os_factor)
        self.am_length_symbols = int(config['system']['am_length_symbols'])

        tx = config.get('tx', {})
        self.baud_rate = float(config['system']['baud_rate'])

        # DAC
        dac_cfg = tx.get('dac', {})
        self.enob = dac_cfg.get('physical_bits', 8)
        self.clip_papr = np.random.uniform(dac_cfg.get('clip_papr_db_min'),
                                           dac_cfg.get('clip_papr_db_max'))
        self.inl_sigma = dac_cfg.get('inl_sigma_max', 1.5)
        self.dnl_sigma = dac_cfg.get('dnl_sigma_max', 0.8)

        # Clocking
        clock_cfg = tx.get('clocking', {})
        self.jitter_rms = np.random.uniform(clock_cfg.get('random_jitter_rms_fs_min'),
                                            clock_cfg.get('random_jitter_rms_fs_max')) * 1e-15

        # IQ Imbalance
        iq_cfg = tx.get('iq_imbalance', {})
        self.iq_gain_db = np.random.normal(iq_cfg.get('gain_db_mu'),
                                           iq_cfg.get('gain_db_sigma_max'))
        self.iq_phase_deg = np.random.normal(iq_cfg.get('phase_deg_mu'),
                                             iq_cfg.get('phase_deg_sigma_max'))
        #
        # self.iq_gain_db = np.random.normal(iq_cfg.get('gain_db_mu', 0.0),
        #                                    iq_cfg.get('gain_db_sigma_max', 0.8))
        # self.iq_phase_deg = np.random.normal(iq_cfg.get('phase_deg_mu', 0.0),
        #                                      iq_cfg.get('phase_deg_sigma_max', 4.0))

        # Modulator
        mod_cfg = tx.get('modulator', {})
        self.vpp_vpi = np.random.uniform(mod_cfg.get('vpp_to_vpi_ratio_min'),
                                         mod_cfg.get('vpp_to_vpi_ratio_max'))
        self.bias_drift = np.random.uniform(mod_cfg.get('bias_drift_pct_min'),
                                            mod_cfg.get('bias_drift_pct_max')) / 100.0
        er_db = np.random.uniform(mod_cfg.get('extinction_ratio_db_min'),
                                  mod_cfg.get('extinction_ratio_db_max'))

        # self.vpp_vpi = np.random.uniform(mod_cfg.get('vpp_to_vpi_ratio_min', 0.4),
        #                                  mod_cfg.get('vpp_to_vpi_ratio_max', 0.9))
        # self.bias_drift = np.random.uniform(mod_cfg.get('bias_drift_pct_min', -8.0),
        #                                     mod_cfg.get('bias_drift_pct_max', 8.0)) / 100.0
        # er_db = np.random.uniform(mod_cfg.get('extinction_ratio_db_min', 16.0),
        #                           mod_cfg.get('extinction_ratio_db_max', 28.0))
        #
        self.er_amp = 10 ** (-er_db / 20.0)

        # RF Driver / MPM
        rf_cfg = tx.get('rf_driver', {})
        self.mpm_k = rf_cfg.get('poly_order_k')
        self.mpm_m = rf_cfg.get('memory_depth_m')
        sigma_nl = rf_cfg.get('non_linear_coefficients_sigma')

        # rf_cfg = tx.get('rf_driver', {})
        # self.mpm_k = rf_cfg.get('poly_order_k', 5)
        # self.mpm_m = rf_cfg.get('memory_depth_m', 4)
        # sigma_nl = rf_cfg.get('non_linear_coefficients_sigma', 0.15)

        # Generate MPM Coefficient Matrix (Memory Depth x Odd Orders)
        self.orders = np.arange(1, self.mpm_k + 1, 2)
        self.mpm_coeffs = np.random.normal(0, sigma_nl, size=(self.mpm_m, len(self.orders)))
        self.mpm_coeffs[0, 0] = 1.0  # Main instantaneous linear tap must be normalized to ~1.0

        # --- pre-calculate hardware tables & filters
        self._generate_dac_grid()

        # Bessel Filter Cutoff (properly normalized against Simulation Nyquist)
        cutoff_ratio = 1.9 / self.os_factor  # 1 is Simulation Nyquist
        span = 64  # number of taps at simulation frequency
        self.tx_afe_taps = generate_bessel_taps(cutoff_ratio=cutoff_ratio, span=span)

        _, _, self.bits_per_ch = get_qam_lut(self.modulation)

        # Determine the amount of padding symbols at the start and end of each frame
        if self.config['channel']['mode'] == 'ideal':
            self.padding_UI_len = 64
        else:
            dominant_UI_delay = self.config['channel']['absolute_delay_ui_max']
            # add 2*dominant delay , rounded up to a power of 2 of zeros at the end of the payload
            self.padding_UI_len = 2 ** np.ceil(np.log2(2 * dominant_UI_delay)).astype(int)

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

    def transmit_batch(self, batch_size, payload_bits):
        """Executes the full pipeline for a batched row tensor."""

        payload_symbols = payload_bits // (2 * self.bits_per_ch)

        # Calculate padding at the start and end of each frame;  this is based on
        # the maximum dominant delay in the channel
        # padding is BPSK; pad_lenT
        plen = self.padding_UI_len  # bpsk symbols (i.e. bits)
        pad_v = np.random.choice([-1.0, 1.0], size=(1, plen)).astype(np.complex64)
        pad = np.tile(pad_v, (batch_size, 1))
        pre_pad_len = plen // 4
        pre_pad = pad[:, :pre_pad_len]
        post_pad = pad[:, pre_pad_len:]

        # Generate 256-symbol AM and Payload
        am_ref_v = generate_walsh_hadamard_am(length=self.am_length_symbols, lane_idx=0)
        # Apply standard Wi-Fi normalization to QPSK AM to match payload power
        am_ref_v *= (1.0 / np.sqrt(2))
        am_ref = np.tile(am_ref_v, (batch_size, 1))  # broadcast to batch size

        payload = self._step1_generate_payload(batch_size, payload_symbols)

        # Concatenate in the 100 Gbd digital domain
        syms = np.concatenate((pre_pad, am_ref, payload, post_pad), axis=1)

        # DAC Digital Domain (e.g. 200 GSa/s)
        dac_out = self._step2_dac_digital_domain(syms)

        # Electrical Analog (Simulation) Domain (e.g. 800 GSa/s)
        I_rf, Q_rf = self._step3_analog_domain(dac_out)

        # Electro-Optic Domain (e.g. 800 GSa/s Complex Baseband)
        tx_optical_field = self._step4_modulator(I_rf, Q_rf)

        if False:
            # sig_to_show = I_rf + 1j * Q_rf
            sig_to_show = tx_optical_field
            os_factor = 8
            s0 = 150
            s1 = 6000
            for ix in range(os_factor):
                oqam_eye(sig_to_show[0, s0 + ix:s1], os_factor)

        return am_ref_v, tx_optical_field

    def _step1_generate_payload(self, batch_size, payload_symbols):
        """Generates random payload and maps to discrete QAM voltage levels."""
        lut, k_mod, _ = get_qam_lut(self.modulation)

        i_idx = np.random.randint(0, 2 ** self.bits_per_ch, size=(batch_size, payload_symbols))
        q_idx = np.random.randint(0, 2 ** self.bits_per_ch, size=(batch_size, payload_symbols))

        I_payload = lut[i_idx] * k_mod
        Q_payload = lut[q_idx] * k_mod
        payload = I_payload + 1j * Q_payload
        return payload

    def _step2_dac_digital_domain(self, syms):
        """100Gbd to 200GSa/s (2x OS), OQAM Shift, and DAC Quantization."""

        # Upsample to 2x (200 GSa/s)
        syms_2x = np.repeat(syms, 2, axis=1)
        I_2x = np.real(syms_2x)
        Q_2x = np.imag(syms_2x)

        # Offset QAM (Shift Q by exactly 1 sample at 2x rate = T/2)
        padding = np.zeros((I_2x.shape[0], 1))
        I_oqam = np.concatenate((I_2x, padding), axis=1)
        Q_oqam = np.concatenate((padding, Q_2x), axis=1)

        # PAPR Clipping
        clip_linear = 10 ** (self.clip_papr / 20.0)

        # Quantization (Nearest-neighbor mapping to pre-calculated INL/DNL grid)
        def quantize_to_grid(x):
            x_norm = np.clip(x / clip_linear, -1, 1)
            # Find closest index in the dac_grid using broadcasting
            idx = np.abs(x_norm[..., np.newaxis] - self.dac_grid).argmin(axis=-1)
            return self.dac_grid[idx] * clip_linear

        I_quantized = quantize_to_grid(I_oqam)
        Q_quantized = quantize_to_grid(Q_oqam)

        quantized_syms = I_quantized + 1j * Q_quantized

        return quantized_syms

    def _step3_analog_domain(self, dac_out):
        """200GSa/s to 800GSa/s (4x OS), Filtering, and Volterra PA Non-linearities."""
        analog_os = self.os_factor // 2
        I_2x = np.real(dac_out)
        Q_2x = np.imag(dac_out)

        # Continuous Time Projection (e.g. 800 GSa/s)
        I_8x = np.repeat(I_2x, analog_os, axis=1).astype(np.float32)
        Q_8x = np.repeat(Q_2x, analog_os, axis=1).astype(np.float32)

        debug_enable_AM_jitter = True
        print(f"jitter_rms: {self.jitter_rms}")
        if debug_enable_AM_jitter:
            # Jitter Injection (Taylor Series Derivative Approximation)
            dt_jitter_I = np.random.normal(0, self.jitter_rms, I_8x.shape)
            dt_jitter_Q = np.random.normal(0, self.jitter_rms, Q_8x.shape)

            I_8x += np.gradient(I_8x, axis=1) * (dt_jitter_I / self.T_ana)
            Q_8x += np.gradient(Q_8x, axis=1) * (dt_jitter_Q / self.T_ana)

        # Tx AFE Linear Low-Pass Filtering
        I_filt = fft_convolve_same(I_8x, self.tx_afe_taps)
        Q_filt = fft_convolve_same(Q_8x, self.tx_afe_taps)

        # RF Driver Memory Polynomial (MPM)
        I_nl = np.zeros_like(I_filt)
        Q_nl = np.zeros_like(Q_filt)

        # Generalized Volterra loop
        enable_Volterra = True
        if enable_Volterra:  # debug
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
        else:
            I_nl = I_filt
            Q_nl = Q_filt

        return I_nl, Q_nl

    def _step4_modulator(self, I_nl, Q_nl):
        """Apply Analog I/Q Imbalances and Electro-Optic Modulator sine curve."""
        # I/Q Imbalance (Gain and Phase applied in continuous domain)
        gain_lin = 10 ** (self.iq_gain_db / 20.0)
        phase_rad = np.radians(self.iq_phase_deg)

        # Gain mismatch (I is slightly larger, Q is slightly smaller)
        I_imb = I_nl * np.sqrt(gain_lin)

        # Phase mismatch (Q bleeds into I)
        Q_imb = (Q_nl * np.cos(phase_rad) - I_nl * np.sin(phase_rad)) / np.sqrt(gain_lin)

        # Mach-Zehnder Modulator Transfer Function
        # Vpi normalized sine response: sin(V * Vpp/Vpi * pi/2 + bias_drift)
        theta_I = I_imb * self.vpp_vpi * (np.pi / 2) + (self.bias_drift * np.pi)
        theta_Q = Q_imb * self.vpp_vpi * (np.pi / 2) + (self.bias_drift * np.pi)

        # Add Extinction Ratio floor (finite null depth)
        I_opt = np.sin(theta_I) + self.er_amp
        Q_opt = np.sin(theta_Q) + self.er_amp
        modulator_out = (I_opt + 1j * Q_opt).astype(np.complex64)

        return modulator_out
