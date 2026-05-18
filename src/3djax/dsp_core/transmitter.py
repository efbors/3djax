import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


class Transmitter:
    def __init__(self, config):
        self.config = config
        self.baud_rate = float(config['system']['baud_rate'])
        self.simulation_duration = config['system']['target_simulation_duration']
        self.start_delay = config['tx']['start_delay']
        self.os_factor = int(config['system']['os_factor'])
        self.tx_ppm = config['tx']['frequency_offset_ppm']

    def generate_signal(self):
        """
                Generate random PAM4 symbols, upsample by self.os_factor,
                apply a smooth start ramp, insert physical start delay,
                and apply pulse shaping.
        """

        # The tx waveform does not start on sample zero;  there is a delay specified by start_delay_s
        # this delay is partitioned into two: an integer number of analog samples and a fractional one
        sample_rate = self.baud_rate * self.os_factor  # e.g. 100e9 * 8 = 800e9 samples/s
        sample_period = 1.0 / sample_rate  # e.g. 1.25 ps/sample
        symbol_period = 1.0 / self.baud_rate

        # Convert the desired start delay into samples (may be fractional)
        delay_samples = self.start_delay / sample_period
        integer_delay_samples = int(np.floor(delay_samples))
        fractional_delay_samples = delay_samples - integer_delay_samples

        # Calculate exact number of symbols needed to fill the duration
        num_symbols = int(np.ceil(self.simulation_duration / symbol_period)) + int(
            np.ceil(delay_samples / self.os_factor))

        # Generate random PAM4 symbols
        symbols = np.random.choice([-3.0, -1.0, 1.0, 3.0], size=num_symbols).astype(np.float32)

        # Upsample Zero order Hold (ZOH)
        tx_waveform = np.repeat(symbols, self.os_factor).astype(np.float32)

        # Apply smooth start ramp (Half-Hann window over 16 symbols)
        ramp_symbols = 16
        ramp_samples = ramp_symbols * self.os_factor
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples)).astype(np.float32)
        tx_waveform[:ramp_samples] *= ramp

        # Insert integer delay (in analog samples)
        if integer_delay_samples > 0:
            pad_zeros = np.zeros(integer_delay_samples, dtype=np.float32)
            tx_waveform = np.concatenate((pad_zeros, tx_waveform))

        # pulse shape (remove images)
        tx_upsampled_ps = gaussian_filter1d(tx_waveform, sigma=1.0)

        # Apply transmitter frequency offset
        # tx_ppm < 0  -> transmitter is slower
        # tx_ppm > 0  -> transmitter is faster
        eps = self.tx_ppm * 1e-6  # fractional frequency error

        # jitter: Convert UI to analog samples
        jitter_rms_ui = float(self.config['tx'].get('jitter_rms_ui', 0.02))
        jitter_rms_samples = jitter_rms_ui * self.os_factor

        N = len(tx_upsampled_ps)
        n = np.arange(N, dtype=np.float64)

        # -----------------------------------------------------------------
        # Simulate 802.3dj Realistic Correlated Jitter
        # -----------------------------------------------------------------
        # Generate pure white Gaussian phase noise
        raw_jitter = np.random.normal(0.0, jitter_rms_samples, size=N)

        # Smooth it out ( A real PLL cannot change phase instantly)
        # A sigma of os_factor smooths the phase walk over roughly 1 UI.
        smoothed_jitter = gaussian_filter1d(raw_jitter, sigma=self.os_factor)

        # Filtering reduces the overall RMS power. We must rescale it
        # back up to exactly match your targeted 0.02 UI RMS jitter.
        current_rms = np.std(smoothed_jitter)
        if current_rms > 0:
            smoothed_jitter = smoothed_jitter * (jitter_rms_samples / current_rms)

        # -----------------------------------------------------------------
        # The Master Timing Equation
        # -----------------------------------------------------------------
        # n: Nominal sample index
        # n * eps: Frequency offset (PPM drift)
        # - fractional_delay_samples: Sub-sample static shift
        # - smoothed_jitter: Dynamic phase noise (Jitter)
        #
        # Note: SUBTRACT delays. To shift a waveform to the right (later in time),
        # we evaluate the unshifted spline at an earlier point.

        t = n * (1.0 + eps) - fractional_delay_samples - smoothed_jitter

        # Cubic spline interpolator over the original samples
        cs = CubicSpline(n, tx_upsampled_ps, bc_type='natural')

        # Evaluate only where interpolation is valid
        tx_analog = np.zeros_like(tx_upsampled_ps, dtype=np.float32)
        valid = (t >= 0.0) & (t <= (N - 1))
        tx_analog[valid] = cs(t[valid]).astype(np.float32)
        return symbols, tx_analog
