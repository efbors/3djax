import numpy as np
from utils.fft_convolve_same import fft_convolve_same
from utils.bessel import generate_bessel_taps
from utils.qam_mapper import get_qam_lut
from utils.diagnostics import oqam_eye, find_best_phase_QAM16, plot_time_domain


# Assumes generate_bessel_taps and fft_convolve_same are available

class RxAFE:
    def __init__(self, config):
        self.config = config
        self.baud_rate = float(config['system']['baud_rate'])
        self.os_analog = int(config['system']['os_factor'])  # Analog OS = 8 (800 GSa/s)
        self.os_digital = 2  # Digital OS = 2 (200 GSa/s)
        self.downsample_ratio = self.os_analog // self.os_digital

        self.payload_bits = self.config['system']['payload_bits']
        self.modulation = config['system']['modulation']
        _, _, self.bits_per_ch = get_qam_lut(self.modulation)
        self.payload_symbols = self.payload_bits // (2 * self.bits_per_ch)

        self.am_type = config['system']['am_type']
        self.am_symbols = self.config['system'].get('am_length_symbols')

        self.fs_analog = self.baud_rate * self.os_analog
        self.fs_digital = self.baud_rate * self.os_digital
        self.T_dig = 1.0 / self.fs_digital

        rx_cfg = config.get('rx', {})
        afe = rx_cfg.get('afe', {})
        adc = rx_cfg.get('adc', {})

        # 0. Parse clocking from configuration
        clock_cfg = rx_cfg.get('clocking', {})
        # 1. Uniformly sample an RMS jitter value in UI for this batch instance
        self.jitter_rms_ui = np.random.uniform(clock_cfg.get('jitter_rms_ui_min'),
                                               clock_cfg.get('jitter_rms_ui_max'))

        # 2. Convert Unit Intervals (UI) to physical seconds
        # Fundamental relation: 1 UI = 1 / Baud Rate
        self.T_sym = 1.0 / self.baud_rate
        self.jitter_sec = self.jitter_rms_ui * self.T_sym

        # 1. AFE / TIA Parameters
        self.agc_target = afe.get('agc_target_amplitude')

        # Uniformly slide the bandwidth to create dataset diversity
        bw_ghz = np.random.uniform(afe.get('3db_bandwidth_ghz_min'),
                                   afe.get('3db_bandwidth_ghz_max'))

        # Calculate digital normalized cutoff (1.0 = Simulation Nyquist)
        # sim_nyquist_ghz = (self.fs_analog / 1e9) / 2.0
        # cutoff_ratio = bw_ghz / sim_nyquist_ghz
        cutoff_ratio = 2 / self.os_analog

        span = 64  # number of taps at simulation frequency

        # Generate anti-aliasing filter at 800 GSa/s (Hardcoded 4th-order)
        self.rx_afe_taps = generate_bessel_taps(cutoff_ratio=cutoff_ratio, span=span)

        # 2. ADC Parameters
        self.enob = adc.get('physical_bits', 8)
        self.clip_linear = 10 ** (np.random.uniform(adc.get('clip_papr_db_min'),
                                                    adc.get('clip_papr_db_max')) / 20.0)
        self.inl_sigma = adc.get('inl_sigma_max')
        self.dnl_sigma = adc.get('dnl_sigma_max')

        # 3. ADC Time-Interleaving (TI) Parameters
        self.L = adc.get('ti_interleave_factor', 4)
        gain_miss = adc.get('ti_gain_mismatch_pct_max') / 100.0
        skew_fs = adc.get('ti_skew_fs_max') * 1e-15
        dc_miss = adc.get('ti_dc_offset_mv_max') / 1000.0

        # Generate TI mismatch arrays (independent for I and Q physical ADCs)
        self.ti_gain_I = np.random.normal(1.0, gain_miss, self.L)
        self.ti_gain_Q = np.random.normal(1.0, gain_miss, self.L)

        self.ti_skew_I = np.random.normal(0.0, skew_fs, self.L)
        self.ti_skew_Q = np.random.normal(0.0, skew_fs, self.L)

        self.ti_dc_I = np.random.normal(0.0, dc_miss, self.L)
        self.ti_dc_Q = np.random.normal(0.0, dc_miss, self.L)

        # Pre-calculate Independent I/Q INL/DNL Grids
        self.dac_grid_I = self._generate_quantization_grid()
        self.dac_grid_Q = self._generate_quantization_grid()

        # Generate the split 256-point taper (Hann window)
        # np.hanning(256) fades from 0 to 1 over 128 samples, then 1 to 0 over 128 samples.
        taper = np.hanning(256)
        self.fade_in = taper[:128]
        self.fade_out = taper[128:]

    def _generate_quantization_grid(self):
        """Generates static INL/DNL voltage map for an ADC."""
        levels = 2 ** self.enob
        ideal_steps = np.linspace(-1, 1, levels)
        dnl = np.random.normal(0, self.dnl_sigma / levels, levels)
        dnl[0] = 0
        inl = np.cumsum(dnl)
        inl -= np.linspace(inl[0], inl[-1], levels)
        bow = np.sin(np.linspace(0, np.pi, levels)) * (self.inl_sigma / levels)
        return ideal_steps + inl + bow

    def process(self, am_ref, rx_analog):
        """

        :param am_ref: alignment marker vector, normalized, same for all the frames in the batch;

        Processes the complex 800 GSa/s waveform through the Rx physics
        and downsamples to a quantized 200 GSa/s complex tensor.
        """

        # -- rx clock jitter (800 GSa/s Continuous Domain)
        # Apply Taylor series derivative approximation for complex timing jitter
        dt_jitter_I = np.random.normal(0, self.jitter_sec, rx_analog.shape)
        dt_jitter_Q = np.random.normal(0, self.jitter_sec, rx_analog.shape)

        # Apply to the real and imaginary parts independently
        rx_I_jit = np.real(rx_analog) + np.gradient(np.real(rx_analog), axis=-1) * (
                dt_jitter_I * self.fs_analog)
        rx_Q_jit = np.imag(rx_analog) + np.gradient(np.imag(rx_analog), axis=-1) * (
                dt_jitter_Q * self.fs_analog)
        rx_jittered = rx_I_jit + 1j * rx_Q_jit

        # -- rx afe anti-aliasing filter (e.g. 800 gsa/s)
        # Apply filter independently to real and imag parts of the jittered signal
        rx_I = fft_convolve_same(np.real(rx_jittered), self.rx_afe_taps)
        rx_Q = fft_convolve_same(np.imag(rx_jittered), self.rx_afe_taps)
        rx_filt = rx_I + 1j * rx_Q

        # AGC - Block-based RMS scaling
        # Calculates the RMS power of the batch and scales it to the target
        rms_power = np.sqrt(np.mean(np.abs(rx_filt) ** 2, axis=-1, keepdims=True))
        agc_gain = self.agc_target / np.clip(rms_power, 1e-6, None)
        rx_agc_sim = rx_filt * agc_gain  # at Fsim

        # Apply the taper to the batch boundaries
        rx_agc_sim[:, :128] *= self.fade_in
        rx_agc_sim[:, -128:] *= self.fade_out

        # Find timing alignment in the Fsim domain (e.g.  800Gs/s)
        am_start_indices = self._detect_timing(self.os_analog, am_ref, rx_agc_sim)

        # The alignment indices are in the Fsim domain;  rotate by the fractional
        # amount to land each signal aligned on the Fadc grid ( less than Fsim//Fadc)
        fractional_am_shifts = am_start_indices % self.downsample_ratio

        # Build the decimation extraction grid
        # base_grid is [0, 4, 8, 12, ...]
        N_fsim = rx_agc_sim.shape[1]

        # Perform the full floating-point alignment in one FFT operation
        # am_start_fp is the precise fractional (on the Fadc) floating point peak (e.g., 3.7)
        # Shift amount is negative to pull the wave left towards the integer grid zero-phase
        shift_amount = -fractional_am_shifts

        freqs = np.fft.fftfreq(N_fsim)
        phase_ramp = np.exp(-1j * 2 * np.pi * freqs * shift_amount[:, np.newaxis])

        rx_fft = np.fft.fft(rx_agc_sim, axis=1)
        rx_agc_aligned = np.fft.ifft(rx_fft * phase_ramp, axis=1)

        # Decimate directly from the zero-phase grid
        N_fadc = N_fsim // self.downsample_ratio
        extraction_grid = np.arange(N_fadc) * self.downsample_ratio

        # Broadcast the 1D extraction grid across the 2D batch
        rx_agc = rx_agc_aligned[:, extraction_grid]
        am_start_indices_fx = (am_start_indices // self.downsample_ratio).astype(np.int32)

        if False:
            # Check rx signal before ADC impairments
            # Plot the AM ZC
            sig_to_show = rx_agc[0]
            s0 = am_start_indices_fx[0]
            s1 = s0 + 4000
            oqam, os_factor = False, 2
            oqam_eye(sig_to_show[s0 - 1:s1], os_factor, oqam)
            oqam_eye(sig_to_show[s0:s1], os_factor, oqam)
            oqam_eye(sig_to_show[s0 + 1:s1], os_factor, oqam)
            # Plot the QAM16 payload
            s0 = am_start_indices_fx[0] + 4006
            s1 = s0 + 900
            oqam, os_factor = True, 2
            oqam_eye(sig_to_show[s0 - 1:s1], os_factor, oqam)
            oqam_eye(sig_to_show[s0:s1], os_factor, oqam, title="Timing Accurate (200 GSa/s)")
            oqam_eye(sig_to_show[s0 + 1:s1], os_factor, oqam)

        I_agc = np.real(rx_agc)
        Q_agc = np.imag(rx_agc)

        # Apply ti mismatches (Derivative approximation for Skew)
        # Calculate gradients for skew timing offsets
        grad_I = np.gradient(I_agc, axis=-1)
        grad_Q = np.gradient(Q_agc, axis=-1)

        # Apply the L-length repeating mismatch arrays via vectorized slicing
        for i in range(self.L):
            # Apply Gain and DC Offset
            I_agc[:, i::self.L] = (I_agc[:, i::self.L] * self.ti_gain_I[i]) + self.ti_dc_I[i]
            Q_agc[:, i::self.L] = (Q_agc[:, i::self.L] * self.ti_gain_Q[i]) + self.ti_dc_Q[i]

            # Apply Skew (Timing offset via gradient Taylor series)
            I_agc[:, i::self.L] += grad_I[:, i::self.L] * (self.ti_skew_I[i] / self.T_dig)
            Q_agc[:, i::self.L] += grad_Q[:, i::self.L] * (self.ti_skew_Q[i] / self.T_dig)

        # O(1) Quantizer to the INL/DNL grid
        def quantize_hybrid(x, grid):
            levels = len(grid)

            # Normalize and clip
            x_norm = np.clip(x / self.clip_linear, -1.0, 1.0)

            # Map continuous voltage directly to an integer index (the ADC Code) - O(1) time
            idx = np.round((x_norm + 1.0) / 2.0 * (levels - 1)).astype(np.int32)

            # Direct lookup into the non-linear physical grid
            return grid[idx] * self.clip_linear

        I_out = quantize_hybrid(I_agc, self.dac_grid_I)
        Q_out = quantize_hybrid(Q_agc, self.dac_grid_Q)

        # Recombine into complex 200 GSa/s output
        rx_adc_out = (I_out + 1j * Q_out).astype(np.complex64)

        if False:
            # Plot the output of the Receiver chain at Fadc=100;  timing is aligned
            # Plot the AM ZC
            sig_to_show = rx_adc_out[0]
            s0 = am_start_indices_fx[0]
            s1 = s0 + 4000
            oqam, os_factor = False, 2
            oqam_eye(sig_to_show[s0 - 1:s1], os_factor, oqam)
            oqam_eye(sig_to_show[s0:s1], os_factor, oqam)
            oqam_eye(sig_to_show[s0 + 1:s1], os_factor, oqam)
            # Plot the QAM16 payload
            s0 = am_start_indices_fx[0] + 4006
            s1 = s0 + 900
            oqam, os_factor = True, 2
            oqam_eye(sig_to_show[s0 - 1:s1], os_factor, oqam)
            oqam_eye(sig_to_show[s0:s1], os_factor, oqam, title="Timing Accurate (200 GSa/s)")
            oqam_eye(sig_to_show[s0 + 1:s1], os_factor, oqam)

        return rx_adc_out, am_start_indices_fx

    def _detect_timing(self, os_factor, am_ref, rx_agc):
        """
        Detects the starting index of the alignment marker for each row in the batch.
        """
        batch_size = rx_agc.shape[0]

        # Upsample the reference by inserting (os_factor - 1) zeros between symbols
        am_len = am_ref.shape[-1]
        filter_len = am_len * os_factor

        am_upsampled = np.zeros(am_len * os_factor, dtype=np.complex64)
        am_upsampled[::os_factor] = am_ref

        # Broadcast to match the batch dimension
        am_broadcast = np.broadcast_to(am_upsampled, (batch_size, filter_len))

        # Create the matched filter
        matched_filter = np.conj(am_broadcast[:, ::-1])

        # Perform the cross-correlation via the custom fft_convolve_same
        corr = fft_convolve_same(rx_agc, matched_filter)

        # Extract the index of the maximum correlation peak per row
        peak_indices = np.argmax(np.abs(corr), axis=-1)

        # Clip the integer peaks to prevent IndexError.
        max_idx = corr.shape[-1] - 1
        m_safe = np.clip(peak_indices, 1, max_idx - 1)

        # --- Parabolic interpolation of the peak abscissa
        # Extract the peak and its immediate neighbors across the batch
        row_idx = np.arange(batch_size)
        y_minus1 = np.abs(corr[row_idx, m_safe - 1])
        y_0 = np.abs(corr[row_idx, m_safe])
        y_plus1 = np.abs(corr[row_idx, m_safe + 1])

        # Calculate the fractional offset (delta)
        # A small epsilon (1e-12) is added to the denominator to prevent a
        # ZeroDivisionError in the unlikely event of a perfectly flat peak.
        numerator = y_plus1 - y_minus1
        denominator = 2 * y_0 - y_plus1 - y_minus1 + 1e-12
        delta = 0.5 * (numerator / denominator)

        # Compute the final floating-point alignment indices
        peak_indices_fp = m_safe + delta

        # Shift the peak back to the true start of the frame (Index 0 of the AM)
        # The 'same' convolution delays the peak by exactly half the filter length
        center_offset = filter_len // 2
        start_indices = peak_indices_fp - center_offset

        if False:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(np.abs(corr[0]), 'b', label='Magnitude (Abs)', alpha=0.5)
            plt.title('correlation with ZC AM')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend()
            plt.show(block=False)

        return start_indices
