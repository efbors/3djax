import numpy as np
from utils.fft_convolve_same import fft_convolve_same
from utils.bessel import generate_bessel_taps

# Assumes generate_bessel_taps and fft_convolve_same are available

class ReceiverAnalogFrontEnd:
    def __init__(self, config):
        self.config = config
        self.baud_rate = float(config['system']['baud_rate'])
        self.os_analog = int(config['system']['os_factor'])  # Analog OS = 8 (800 GSa/s)
        self.os_digital = 2  # Digital OS = 2 (200 GSa/s)
        self.downsample_ratio = self.os_analog // self.os_digital

        self.fs_analog = self.baud_rate * self.os_analog
        self.fs_digital = self.baud_rate * self.os_digital
        self.T_dig = 1.0 / self.fs_digital

        rx_cfg = config.get('rx', {})
        afe = rx_cfg.get('afe', {})
        adc = rx_cfg.get('adc', {})

        # 1. AFE / TIA Parameters
        # 1. AFE / TIA Parameters
        self.agc_target = afe.get('agc_target_amplitude', 0.8)

        # Uniformly slide the bandwidth to create dataset diversity
        bw_ghz = np.random.uniform(afe.get('3db_bandwidth_ghz_min', 45.0),
                                   afe.get('3db_bandwidth_ghz_max', 75.0))

        # Calculate digital normalized cutoff (1.0 = Simulation Nyquist)
        sim_nyquist_ghz = (self.fs_analog / 1e9) / 2.0
        cutoff_ratio = bw_ghz / sim_nyquist_ghz

        span = 64  # number of taps at simulation frequency

        # Generate anti-aliasing filter at 800 GSa/s (Hardcoded 4th-order in utility)
        self.rx_afe_taps = generate_bessel_taps(cutoff_ratio=cutoff_ratio, span=span)

        # 2. ADC Parameters
        self.enob = adc.get('physical_bits', 8)
        self.clip_linear = 10 ** (np.random.uniform(adc.get('clip_papr_db_min', 6.0),
                                                    adc.get('clip_papr_db_max', 10.0)) / 20.0)
        self.inl_sigma = adc.get('inl_sigma_max', 1.5)
        self.dnl_sigma = adc.get('dnl_sigma_max', 0.8)

        # 3. ADC Time-Interleaving (TI) Parameters
        self.L = adc.get('ti_interleave_factor', 4)
        gain_miss = adc.get('ti_gain_mismatch_pct_max', 3.0) / 100.0
        skew_fs = adc.get('ti_skew_fs_max', 150.0) * 1e-15
        dc_miss = adc.get('ti_dc_offset_mv_max', 5.0) / 1000.0

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

    def process(self, rx_analog_800g):
        """
        Processes the complex 800 GSa/s waveform through the Rx physics
        and downsamples to a quantized 200 GSa/s complex tensor.
        """
        # 0. RX CLOCK JITTER (800 GSa/s Continuous Domain)
        # Apply Taylor series derivative approximation for complex timing jitter
        dt_jitter_I = np.random.normal(0, self.jitter_sec, rx_analog_800g.shape)
        dt_jitter_Q = np.random.normal(0, self.jitter_sec, rx_analog_800g.shape)

        # Apply to the real and imaginary parts independently
        rx_I_jit = np.real(rx_analog_800g) + np.gradient(np.real(rx_analog_800g), axis=-1) * (
                    dt_jitter_I * self.fs_analog)
        rx_Q_jit = np.imag(rx_analog_800g) + np.gradient(np.imag(rx_analog_800g), axis=-1) * (
                    dt_jitter_Q * self.fs_analog)
        rx_jittered = rx_I_jit + 1j * rx_Q_jit

        # 1. RX AFE ANTI-ALIASING FILTER (800 GSa/s)
        # Apply filter independently to real and imag parts of the jittered signal
        rx_I = fft_convolve_same(np.real(rx_jittered), self.rx_afe_taps)
        rx_Q = fft_convolve_same(np.imag(rx_jittered), self.rx_afe_taps)
        rx_filt = rx_I + 1j * rx_Q

        # 2. AUTOMATIC GAIN CONTROL (Block-based RMS scaling)
        # Calculates the RMS power of the batch and scales it to the target
        rms_power = np.sqrt(np.mean(np.abs(rx_filt) ** 2, axis=-1, keepdims=True))
        agc_gain = self.agc_target / np.clip(rms_power, 1e-6, None)
        rx_agc = rx_filt * agc_gain

        # 3. DOWNSAMPLE TO ADC RATE (200 GSa/s)
        # Slice the array, taking every 4th sample
        rx_200g = rx_agc[:, ::self.downsample_ratio]

        I_200g = np.real(rx_200g)
        Q_200g = np.imag(rx_200g)

        # 4. APPLY TI MISMATCHES (Derivative approximation for Skew)
        # Calculate gradients for skew timing offsets
        grad_I = np.gradient(I_200g, axis=-1)
        grad_Q = np.gradient(Q_200g, axis=-1)

        # Apply the L-length repeating mismatch arrays via vectorized slicing
        for i in range(self.L):
            # Apply Gain and DC Offset
            I_200g[:, i::self.L] = (I_200g[:, i::self.L] * self.ti_gain_I[i]) + self.ti_dc_I[i]
            Q_200g[:, i::self.L] = (Q_200g[:, i::self.L] * self.ti_gain_Q[i]) + self.ti_dc_Q[i]

            # Apply Skew (Timing offset via gradient Taylor series)
            I_200g[:, i::self.L] += grad_I[:, i::self.L] * (self.ti_skew_I[i] / self.T_dig)
            Q_200g[:, i::self.L] += grad_Q[:, i::self.L] * (self.ti_skew_Q[i] / self.T_dig)

        # 5. TIA SATURATION & QUANTIZATION (Separate physical ADCs)
        def quantize(x, grid):
            x_norm = np.clip(x / self.clip_linear, -1, 1)
            idx = np.abs(x_norm[..., np.newaxis] - grid).argmin(axis=-1)
            return grid[idx] * self.clip_linear

        I_out = quantize(I_200g, self.dac_grid_I)
        Q_out = quantize(Q_200g, self.dac_grid_Q)

        # Recombine into complex 200 GSa/s output
        rx_adc_out = (I_out + 1j * Q_out).astype(np.complex64)

        return rx_adc_out